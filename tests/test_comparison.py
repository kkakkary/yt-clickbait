"""Tests for models/comparison.py (FT-3, NFR-1). Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from ytclickbait.models.comparison import compare, scale_on_train


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    videos = pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(n)],
            "channel_id": [f"c{i // 5}" for i in range(n)],
            "log_age_days": rng.normal(size=n),
            "n_tags": 3,
            "desc_len": 100,
            "n_countries": 1,
            "category_name": ["a", "b"] * (n // 2),
            "publish_week": 1,
            "publish_hour_utc": 1,
            "publish_dow_utc": 1,
            "log_views": 2 * signal + rng.normal(0, 0.5, n),
        }
    )
    label = np.where(np.arange(n) < 240, "train", "test")
    split = pd.DataFrame(
        {"video_id": videos["video_id"], "split_group": label, "split_time": label}
    )
    split["channel_baseline_split_time"] = 0.0
    return videos, split, signal


MODELS = {"controls only": [], "rules": ["rules"], "noise": ["noise"]}


def test_informative_block_beats_controls_and_noise_does_not():
    videos, split, signal = _data()
    blocks = {"rules": signal[:, None], "noise": np.random.default_rng(1).normal(size=(300, 3))}
    out = compare(
        videos, split, blocks, models=MODELS, targets=["log_views"], splits=["split_group"]
    )
    r2 = out.set_index("model")["r2"]
    assert r2["rules"] > 0.8 > r2["controls only"]
    assert r2["noise"] < 0.3
    gain = out.set_index("model")["r2_gain_vs_controls"]
    assert gain["controls only"] == 0 and gain["rules"] > 0.5
    assert out["primary"].all()  # split_group is the primary comparison


def test_test_rows_do_not_change_scaling_or_fit():
    videos, split, signal = _data()
    test = (split["split_group"] == "test").to_numpy()
    block = signal[:, None].copy()
    a = compare(videos, split, {"rules": block}, models={"controls only": [], "rules": ["rules"]},
                targets=["log_views"], splits=["split_group"])  # fmt: skip
    block2 = block.copy()
    block2[test] *= 1000  # changes test inputs only: train scaling must not move
    scaled = scale_on_train(block2, split["split_group"] == "train")
    np.testing.assert_allclose(
        scaled[~test], scale_on_train(block, split["split_group"] == "train")[~test]
    )
    assert (a["n_test"] == test.sum()).all()


def test_requires_controls_only_baseline():
    videos, split, signal = _data()
    with pytest.raises(ValueError, match="controls only"):
        compare(videos, split, {"rules": signal[:, None]}, models={"rules": ["rules"]})


def test_same_rows_scored_for_every_model():
    videos, split, signal = _data()
    blocks = {"rules": signal[:, None], "noise": np.zeros((300, 2))}
    out = compare(
        videos, split, blocks, models=MODELS, targets=["log_views"], splits=["split_group"]
    )
    assert out["n_test"].nunique() == 1 and out["n_train"].nunique() == 1


def test_scale_on_train_uses_train_stats():
    x = np.array([[0.0], [2.0], [100.0]])
    z = scale_on_train(x, pd.Series([True, True, False]))
    assert z[0, 0] == pytest.approx(-1.0) and z[1, 0] == pytest.approx(1.0)
