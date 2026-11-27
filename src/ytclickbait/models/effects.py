"""OLS effects of rule features with channel-clustered CIs (FR-7).

Two models per target and scope:
  pooled          one regression across all channels.
  fixed_effects   the same regression after subtracting each channel's mean from every
                  variable (within-creator). A tactic whose effect shrinks or flips sign here
                  was really about which channels use it.

This is inference on the whole dataset, not prediction, so no train/test split is used.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

log = logging.getLogger(__name__)

RULE_FEATURES = [
    "n_chars",
    "n_words",
    "caps_ratio",
    "n_caps_words",
    "n_exclaim",
    "n_question",
    "has_number",
    "starts_with_number",
    "n_emoji",
    "has_brackets",
    "has_pipe_or_colon",
    "has_hashtag",
    "has_second_person",
    "n_clickbait_words",
]
# n_chars and n_words describe length; they are controls for the other features, not tactics.
TACTIC_FEATURES = [f for f in RULE_FEATURES if f not in ("n_chars", "n_words")]
TARGETS = ["log_views", "log_likes_per_view", "log_comments_per_view"]
NUMERIC_CONTROLS = ["log_age_days", "n_tags", "n_countries"]
DUMMY_CONTROLS = ["category_name", "publish_week", "publish_hour_utc", "publish_dow_utc"]

# What counts as "using" a tactic, for the usage rate (caps_ratio is nonzero in almost every title).
USAGE_RULES = {"caps_ratio": lambda s: s >= 0.5}


def usage_rates(rules: pd.DataFrame) -> pd.Series:
    """Share of titles that use each rule feature (value above 0; caps_ratio at least 0.5)."""
    out = {}
    for f in RULE_FEATURES:
        rule = USAGE_RULES.get(f, lambda s: s > 0)
        out[f] = float(rule(rules[f]).mean())
    return pd.Series(out, name="usage_rate")


def build_design(
    videos: pd.DataFrame, rules: pd.DataFrame, drop_dummies: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Rule features + numeric controls + one-hot controls (first level dropped)."""
    if not videos["video_id"].equals(rules["video_id"]):
        raise ValueError("videos and rule features are out of sync; rerun build_features.py")
    parts = [rules[RULE_FEATURES].astype(float).reset_index(drop=True)]
    numeric = videos[NUMERIC_CONTROLS].astype(float).reset_index(drop=True)
    numeric["log_desc_len"] = np.log1p(videos["desc_len"].to_numpy())
    parts.append(numeric)
    cols = [c for c in DUMMY_CONTROLS if c not in drop_dummies]
    parts.append(
        pd.get_dummies(
            videos[cols].astype(str).reset_index(drop=True), drop_first=True, dtype=float
        )
    )
    return pd.concat(parts, axis=1)


def fit_ols(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    features: list[str] = RULE_FEATURES,
    fixed_effects: bool = False,
) -> pd.DataFrame:
    """OLS with channel-clustered standard errors; one row per feature.

    Rows with a missing target are dropped, never treated as zero. With `fixed_effects` each
    variable has its channel mean subtracted first and channels with one video drop out.
    """
    keep = y.notna().to_numpy()
    X, y, g = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), groups[keep]
    g = g.reset_index(drop=True)
    sd = X[features].std()

    if fixed_effects:
        keep = g.map(g.value_counts()).to_numpy() > 1
        X, y, g = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True), g[keep]
        g = g.reset_index(drop=True)
        X = X - X.groupby(g).transform("mean")
        y = y - y.groupby(g).transform("mean")
    X = X.loc[:, X.std() > 1e-10]
    if not fixed_effects:
        X = sm.add_constant(X)

    # Within a category the one-hot controls (hour, week, weekday) can be collinear with each
    # other. That leaves the control coefficients undetermined but not the rule-feature
    # coefficients (checked: rank(all) = rank(features) + rank(controls)), so the warning is
    # silenced for this call only.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sm.tools.sm_exceptions.SingularMatrixWarning)
        result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    ci = result.conf_int(alpha=0.05)
    rows = []
    for f in features:
        if f not in result.params.index:  # constant within the sample: no effect to estimate
            rows.append({"feature": f, "coef": np.nan})
            continue
        b, lo, hi = result.params[f], ci.loc[f, 0], ci.loc[f, 1]
        rows.append(
            {
                "feature": f,
                "coef": b,
                "se": result.bse[f],
                "ci_low": lo,
                "ci_high": hi,
                "p_value": result.pvalues[f],
                "sd": sd[f],
                "std_coef": b * sd[f],
                "std_ci_low": lo * sd[f],
                "std_ci_high": hi * sd[f],
                "pct_effect": 100 * (np.exp(b) - 1),
                "n": int(len(y)),
                "n_channels": int(g.nunique()),
                "r2": float(result.rsquared),
            }
        )
    return pd.DataFrame(rows)


def effects_table(
    videos: pd.DataFrame,
    rules: pd.DataFrame,
    targets: list[str] = TARGETS,
    top_k: int = 10,
) -> pd.DataFrame:
    """Pooled and fixed-effects OLS for every target, overall and in the top-k categories."""
    categories = videos["category_name"].value_counts().index[:top_k].tolist()
    scopes = [("overall", None)] + [(c, c) for c in categories]
    full_X = build_design(videos, rules)

    out = []
    for target in targets:
        for scope, category in scopes:
            if category is None:
                sel = np.ones(len(videos), dtype=bool)
                X = full_X
            else:
                sel = (videos["category_name"] == category).to_numpy()
                X = full_X.loc[
                    sel, [c for c in full_X.columns if not c.startswith("category_name_")]
                ]
            y = videos[target].reset_index(drop=True)[sel].reset_index(drop=True)
            g = videos["channel_id"].reset_index(drop=True)[sel].reset_index(drop=True)
            X = X.reset_index(drop=True)
            for model, fe in [("pooled", False), ("fixed_effects", True)]:
                log.info("OLS %s | %s | %s", target, scope, model)
                res = fit_ols(X, y, g, fixed_effects=fe)
                res.insert(0, "model", model)
                res.insert(0, "scope", scope)
                res.insert(0, "target", target)
                out.append(res)
    return pd.concat(out, ignore_index=True)
