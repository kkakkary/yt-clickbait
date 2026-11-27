"""Tests for models/effects.py (FR-7). Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from ytclickbait.models.effects import RULE_FEATURES, build_design, fit_ols, usage_rates


def _confounded(n_channels=60, per=40, seed=0):
    """Channels that use `n_exclaim` more also get more views, but within a channel the true
    effect of one `!` is -0.10."""
    rng = np.random.default_rng(seed)
    ch = np.repeat(np.arange(n_channels), per)
    taste = rng.normal(size=n_channels)  # channel's style and audience size
    n_exclaim = rng.poisson(np.exp(0.8 * taste[ch]))
    y = 2.0 * taste[ch] - 0.10 * n_exclaim + rng.normal(0, 0.3, len(ch))
    X = pd.DataFrame({"n_exclaim": n_exclaim.astype(float)})
    return X, pd.Series(y), pd.Series(ch)


def test_fixed_effects_recovers_within_channel_effect_pooled_is_biased():
    X, y, g = _confounded()
    pooled = fit_ols(X, y, g, ["n_exclaim"], fixed_effects=False).set_index("feature")
    fe = fit_ols(X, y, g, ["n_exclaim"], fixed_effects=True).set_index("feature")
    assert pooled.loc["n_exclaim", "coef"] > 0  # creator confound makes it look helpful
    assert fe.loc["n_exclaim", "coef"] == pytest.approx(-0.10, abs=0.02)
    assert fe.loc["n_exclaim", "ci_low"] < -0.10 < fe.loc["n_exclaim", "ci_high"]


def test_missing_targets_are_dropped_not_zeroed():
    X, y, g = _confounded()
    y_nan = y.copy()
    y_nan[:50] = np.nan
    a = fit_ols(X, y_nan, g, ["n_exclaim"], fixed_effects=True)
    b = fit_ols(X[50:], y[50:], g[50:], ["n_exclaim"], fixed_effects=True)
    assert a["n"].iloc[0] == b["n"].iloc[0] == len(y) - 50
    assert a["coef"].iloc[0] == pytest.approx(b["coef"].iloc[0])


def test_clustered_se_wider_than_naive_when_channel_effects_exist():
    X, y, g = _confounded()
    import statsmodels.api as sm

    naive = sm.OLS(y, sm.add_constant(X)).fit().bse["n_exclaim"]
    clustered = fit_ols(X, y, g, ["n_exclaim"]).set_index("feature").loc["n_exclaim", "se"]
    assert clustered > naive


def test_constant_feature_gives_nan_not_error():
    X, y, g = _confounded()
    X["has_hashtag"] = 0.0
    out = fit_ols(X, y, g, ["n_exclaim", "has_hashtag"]).set_index("feature")
    assert np.isnan(out.loc["has_hashtag", "coef"])


def test_standardized_effect_is_coef_times_sd():
    X, y, g = _confounded()
    row = fit_ols(X, y, g, ["n_exclaim"]).iloc[0]
    assert row["std_coef"] == pytest.approx(row["coef"] * X["n_exclaim"].std())


def test_usage_rates_and_design_alignment():
    rules = pd.DataFrame({"video_id": ["a", "b", "c", "d"]})
    for f in RULE_FEATURES:
        rules[f] = [0, 0, 1, 1]
    rules["caps_ratio"] = [0.1, 0.6, 0.9, 0.2]
    u = usage_rates(rules)
    assert u["n_exclaim"] == 0.5 and u["caps_ratio"] == 0.5

    videos = pd.DataFrame(
        {
            "video_id": ["a", "b", "c", "d"],
            "log_age_days": [1.0, 2, 3, 4],
            "n_tags": 1,
            "n_countries": 1,
            "desc_len": 10,
            "category_name": ["x", "y", "x", "y"],
            "publish_week": 1,
            "publish_hour_utc": 1,
            "publish_dow_utc": 1,
        }
    )
    assert len(build_design(videos, rules)) == 4
    with pytest.raises(ValueError, match="out of sync"):
        build_design(videos.iloc[::-1].reset_index(drop=True), rules)
