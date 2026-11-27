"""Tests for models/tfidf_model.py (FR-6, NFR-1). Synthetic data only."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ytclickbait.models.tfidf_model import build_controls, fit_ridge, score, top_terms


def _data(n=100, d=5, n_train=80, seed=0):
    rng = np.random.default_rng(seed)
    X = sparse.csr_matrix(rng.normal(size=(n, d)))
    y = pd.Series(X.toarray() @ rng.normal(size=d) + rng.normal(0, 0.1, n))
    return X, y, pd.Series(np.arange(n) < n_train)


def test_ridge_ignores_test_rows():
    X, y, is_train = _data()
    test = ~is_train.to_numpy()
    before = fit_ridge(X, y, is_train)

    X2, y2 = X.toarray(), y.copy()
    X2[test] *= 1000
    y2[test] = 1e6
    after = fit_ridge(sparse.csr_matrix(X2), y2, is_train)

    train = is_train.to_numpy()
    np.testing.assert_allclose(
        before.predict(X[train]), after.predict(sparse.csr_matrix(X2)[train])
    )


def test_ridge_skips_missing_targets():
    X, y, is_train = _data()
    y_nan = y.copy()
    y_nan[:10] = np.nan
    with_nan = fit_ridge(X, y_nan, is_train)
    without = fit_ridge(X, y, is_train & y_nan.notna())
    np.testing.assert_allclose(with_nan.predict(X), without.predict(X))


def test_group_cv_keeps_channels_together():
    X, y, is_train = _data()
    groups = pd.Series(np.arange(100) // 5)
    model = fit_ridge(X, y, is_train, groups=groups)
    assert model.best_params_["alpha"] in model.param_grid["alpha"]


def test_score_known_values():
    class FirstColumn:
        def predict(self, X):
            return X[:, 0].toarray().ravel()

    X = sparse.csr_matrix(np.array([[1.0], [2.0], [3.0], [4.0]]))
    y = pd.Series([1.0, 2.0, 3.0, 5.0])
    s = score(FirstColumn(), X, y, pd.Series([True, True, True, False]))
    assert s["rmse"] == pytest.approx(0.0) and s["r2"] == pytest.approx(1.0) and s["n"] == 3
    assert score(FirstColumn(), X, y, pd.Series(True, index=y.index))["rmse"] == pytest.approx(0.5)
    y_nan = y.copy()
    y_nan[3] = np.nan
    s = score(FirstColumn(), X, y_nan, pd.Series(True, index=y.index))
    assert s["n"] == 3 and s["rmse"] == pytest.approx(0.0)


def _frames(n=40):
    videos = pd.DataFrame(
        {
            "log_age_days": np.arange(n, dtype=float),
            "n_tags": 3,
            "desc_len": 100,
            "n_countries": 1,
            "category_name": ["Music"] * (n - 4) + ["Rare"] * 4,
            "publish_week": 1,
            "publish_hour_utc": 5,
            "publish_dow_utc": 2,
        }
    )
    train = ["train"] * (n - 4) + ["test"] * 4
    split = pd.DataFrame(
        {"split_group": train, "split_time": train, "channel_baseline_split_time": 1.0}
    )
    return videos, split


def test_controls_fit_on_train_only_and_unseen_category_ok():
    videos, split = _frames()
    X, enc = build_controls(videos, split, "split_group")
    assert X.shape[0] == len(videos)
    cats = enc.named_transformers_["cat"].categories_[0]
    assert "Rare" not in cats  # category only in test rows is not learned
    assert X[-4:].shape == (4, X.shape[1])  # unseen category transforms without error


def test_scaler_ignores_test_rows():
    videos, split = _frames()
    a, _ = build_controls(videos, split, "split_group")
    videos2 = videos.copy()
    videos2.loc[videos2.index[-4:], "log_age_days"] = 1e6
    b, _ = build_controls(videos2, split, "split_group")
    np.testing.assert_allclose(a[:-4].toarray(), b[:-4].toarray())


def test_baseline_only_for_split_time():
    videos, split = _frames()
    g, _ = build_controls(videos, split, "split_group")
    t, _ = build_controls(videos, split, "split_time")
    assert t.shape[1] == g.shape[1] + 1


def test_top_terms_finds_planted_word():
    vocab = np.array(["a", "b", "c", "d"])
    coef = np.array([9.0, 9.0, 0.1, 2.0, -3.0, 0.0])  # first 2 are controls
    t = top_terms(coef, vocab, n_controls=2, n=1)
    assert t.loc[t.direction == "positive", "term"].item() == "b"
    assert t.loc[t.direction == "positive", "coef"].item() == 2.0
    assert t.loc[t.direction == "negative", "coef"].item() == -3.0


def test_baseline_fill_ignores_test_rows():
    videos, split = _frames()
    split["channel_baseline_split_time"] = [1.0] * 30 + [None] * 2 + [1.0] * 4 + [None] * 4
    split.loc[split.index[-4:], "channel_baseline_split_time"] = None
    a, _ = build_controls(videos, split, "split_time")
    split2 = split.copy()
    split2.loc[split2.index[-3:], "channel_baseline_split_time"] = 1e6
    b, _ = build_controls(videos, split2, "split_time")
    np.testing.assert_allclose(a[:-4].toarray(), b[:-4].toarray())


def test_top_terms_rejects_vocab_mismatch():
    with pytest.raises(ValueError, match="vocabulary"):
        top_terms(np.zeros(5), np.array(["a", "b"]), n_controls=2)
