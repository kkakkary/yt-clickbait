"""Rank clickbait tactics for the final report (FR-9).

Works on the table written by `models.effects.effects_table` and the usage rates next to it.
A tactic is judged on its 95 % channel-clustered CI, never on the point estimate alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ytclickbait.models.effects import TACTIC_FEATURES

HIGH_USAGE = 0.10  # share of titles that use a tactic for it to count as "common"
ENGAGEMENT_TARGETS = ["log_likes_per_view", "log_comments_per_view"]


def _slice(effects: pd.DataFrame, target: str, scope: str, model: str) -> pd.DataFrame:
    d = effects[
        (effects["target"] == target)
        & (effects["scope"] == scope)
        & (effects["model"] == model)
        & effects["feature"].isin(TACTIC_FEATURES)
    ]
    if d.empty:
        raise ValueError(f"no effects for target={target}, scope={scope}, model={model}")
    return d.copy()


def verdict(ci_low: float, ci_high: float) -> str:
    """'helps' or 'hurts' only when the 95 % CI excludes zero."""
    if np.isnan(ci_low) or np.isnan(ci_high):
        return "not estimated"
    if ci_low > 0:
        return "helps"
    if ci_high < 0:
        return "hurts"
    return "no clear effect"


def rank_tactics(
    effects: pd.DataFrame,
    usage: pd.DataFrame,
    target: str = "log_views",
    scope: str = "overall",
    model: str = "fixed_effects",
    high_usage: float = HIGH_USAGE,
) -> pd.DataFrame:
    """Tactics ordered from most to least helpful (standardized effect = per 1 SD).

    `overused` = common (usage at least `high_usage`) but not clearly helpful: its CI does not
    sit above zero.
    """
    d = _slice(effects, target, scope, model).merge(usage, on="feature", how="left", validate="1:1")
    d["verdict"] = [verdict(lo, hi) for lo, hi in zip(d["ci_low"], d["ci_high"], strict=True)]
    d["common"] = d["usage_rate"] >= high_usage
    d["overused"] = d["common"] & (d["verdict"] != "helps")
    cols = [
        "feature",
        "std_coef",
        "std_ci_low",
        "std_ci_high",
        "pct_effect",
        "usage_rate",
        "verdict",
        "common",
        "overused",
    ]
    return d.sort_values("std_coef", ascending=False)[cols].reset_index(drop=True)


def pooled_vs_within(
    effects: pd.DataFrame, target: str = "log_views", scope: str = "overall"
) -> pd.DataFrame:
    """Pooled vs within-creator effect per tactic, to see which were really about the creator.

    `confounded` = the pooled CI excludes zero but the within-creator one does not, or the signs
    differ. `shrink` = share of the pooled effect that disappears within creators (1 = all of it).
    """
    p = _slice(effects, target, scope, "pooled").set_index("feature")
    f = _slice(effects, target, scope, "fixed_effects").set_index("feature")
    out = pd.DataFrame(
        {
            "pooled": p["std_coef"],
            "within": f["std_coef"],
            "pooled_verdict": [
                verdict(a, b) for a, b in zip(p["ci_low"], p["ci_high"], strict=True)
            ],
            "within_verdict": [
                verdict(a, b) for a, b in zip(f["ci_low"], f["ci_high"], strict=True)
            ],
        }
    )
    out["sign_flip"] = np.sign(out["pooled"]) != np.sign(out["within"])
    out["confounded"] = (out["pooled_verdict"] != "no clear effect") & (
        (out["within_verdict"] == "no clear effect") | out["sign_flip"]
    )
    out["shrink"] = 1 - out["within"].abs() / out["pooled"].abs()
    return out.sort_values("pooled", ascending=False).reset_index(names="feature")


def backlash(
    effects: pd.DataFrame, scope: str = "overall", model: str = "fixed_effects"
) -> pd.DataFrame:
    """Per tactic: views verdict vs likes and comments per view.

    `backlash` = the tactic clearly helps views while clearly hurting likes or comments per view.
    """
    out = None
    for target in ["log_views", *ENGAGEMENT_TARGETS]:
        d = _slice(effects, target, scope, model)
        d[f"verdict_{target}"] = [
            verdict(lo, hi) for lo, hi in zip(d["ci_low"], d["ci_high"], strict=True)
        ]
        d = d.set_index("feature")[["std_coef", f"verdict_{target}"]].rename(
            columns={"std_coef": f"std_{target}"}
        )
        out = d if out is None else out.join(d)
    out["backlash"] = (out["verdict_log_views"] == "helps") & (
        (out["verdict_log_likes_per_view"] == "hurts")
        | (out["verdict_log_comments_per_view"] == "hurts")
    )
    return out.reset_index()
