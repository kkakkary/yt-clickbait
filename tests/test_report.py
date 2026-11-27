"""Tests for report.py (FR-9): ranking, overused and confounded labels."""

import pandas as pd
import pytest

from ytclickbait.report import backlash, pooled_vs_within, rank_tactics, verdict


def _effects():
    rows = []

    def add(target, model, feature, coef, lo, hi):
        rows.append(
            {
                "target": target,
                "scope": "overall",
                "model": model,
                "feature": feature,
                "coef": coef,
                "ci_low": lo,
                "ci_high": hi,
                "std_coef": coef,
                "std_ci_low": lo,
                "std_ci_high": hi,
                "pct_effect": 100 * coef,
            }
        )

    for model in ["pooled", "fixed_effects"]:
        for t in ["log_views", "log_likes_per_view", "log_comments_per_view"]:
            for f in ["n_exclaim", "has_number", "n_emoji", "has_hashtag"]:
                add(t, model, f, 0.0, -0.1, 0.1)
    # rewrite specific cells
    e = pd.DataFrame(rows)

    def setv(target, model, feature, coef, lo, hi):
        m = (e.target == target) & (e.model == model) & (e.feature == feature)
        e.loc[m, ["coef", "std_coef", "ci_low", "std_ci_low", "ci_high", "std_ci_high"]] = [
            coef, coef, lo, lo, hi, hi,
        ]  # fmt: skip

    setv("log_views", "fixed_effects", "n_exclaim", -0.2, -0.3, -0.1)  # common, hurts
    setv("log_views", "fixed_effects", "has_number", 0.0, -0.1, 0.1)  # common, no effect
    setv("log_views", "fixed_effects", "n_emoji", 0.3, 0.2, 0.4)  # rare, helps
    setv("log_views", "pooled", "has_hashtag", 0.5, 0.3, 0.7)  # pooled looks great...
    setv("log_views", "fixed_effects", "has_hashtag", 0.02, -0.1, 0.14)  # ...within: nothing
    setv("log_likes_per_view", "fixed_effects", "n_emoji", -0.2, -0.3, -0.1)
    return e


USAGE = pd.DataFrame(
    {
        "feature": ["n_exclaim", "has_number", "n_emoji", "has_hashtag"],
        "usage_rate": [0.19, 0.38, 0.05, 0.04],
    }
)


def test_verdict_uses_ci_not_point_estimate():
    assert verdict(0.1, 0.3) == "helps"
    assert verdict(-0.3, -0.1) == "hurts"
    assert verdict(-0.1, 0.3) == "no clear effect"
    assert verdict(float("nan"), float("nan")) == "not estimated"


def test_rank_and_overused():
    r = rank_tactics(_effects(), USAGE).set_index("feature")
    assert list(rank_tactics(_effects(), USAGE)["feature"])[0] == "n_emoji"
    assert bool(r.loc["n_exclaim", "overused"]) and bool(r.loc["has_number", "overused"])
    assert not bool(r.loc["n_emoji", "overused"])  # rare and helpful
    assert not bool(r.loc["has_hashtag", "overused"])  # rare


def test_missing_slice_raises():
    with pytest.raises(ValueError, match="no effects"):
        rank_tactics(_effects(), USAGE, target="nope")


def test_pooled_vs_within_flags_confounded_tactic():
    d = pooled_vs_within(_effects()).set_index("feature")
    assert bool(d.loc["has_hashtag", "confounded"])
    assert d.loc["has_hashtag", "shrink"] == pytest.approx(0.96)
    assert not bool(d.loc["n_emoji", "confounded"] and d.loc["n_emoji", "pooled"] != 0)


def test_backlash_flag():
    b = backlash(_effects()).set_index("feature")
    assert bool(b.loc["n_emoji", "backlash"])  # helps views, hurts likes per view
    assert not bool(b.loc["n_exclaim", "backlash"])
