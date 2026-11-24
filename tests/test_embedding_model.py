"""Acceptance tests for FR-11 / FR-12 / NFR-1, written before the code (spec-driven).

They fail until src/ytclickbait/models/embedding_model.py exists with the interface in
docs/SPEC.md §5. The import is inside a fixture so a missing module fails these tests one by
one instead of stopping the whole test run.
"""

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def em():
    return importlib.import_module("ytclickbait.models.embedding_model")


def _data(n=100, d=5, n_train=80, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = pd.Series(X @ rng.normal(size=d) + rng.normal(0, 0.1, n))
    is_train = pd.Series(np.arange(n) < n_train)
    return X, y, is_train


def test_ridge_ignores_test_rows(em):
    """UT-15 (NFR-1, FR-11): changing test rows must not change the fitted model."""
    X, y, is_train = _data()
    test = ~is_train.to_numpy()
    before = em.fit_ridge(X, y, is_train)

    X2, y2 = X.copy(), y.copy()
    X2[test] *= 1000
    y2[test] = 1e6
    after = em.fit_ridge(X2, y2, is_train)

    train = is_train.to_numpy()
    np.testing.assert_allclose(before.predict(X[train]), after.predict(X[train]))


def test_ridge_skips_missing_targets(em):
    """UT-16 (FR-11): engagement is NaN when likes/comments are off; skip those rows."""
    X, y, is_train = _data()
    y_nan = y.copy()
    y_nan[:10] = np.nan
    with_nan = em.fit_ridge(X, y_nan, is_train)
    without = em.fit_ridge(X, y, is_train & y_nan.notna())
    np.testing.assert_allclose(with_nan.predict(X), without.predict(X))


def test_score_known_values(em):
    """UT-17 (FR-11)"""

    class FirstColumn:
        def predict(self, X):
            return X[:, 0]

    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = pd.Series([1.0, 2.0, 3.0, 5.0])

    s = em.score(FirstColumn(), X, y, pd.Series([True, True, True, False]))
    assert s["rmse"] == pytest.approx(0.0)
    assert s["r2"] == pytest.approx(1.0)
    assert s["n"] == 3

    assert em.score(FirstColumn(), X, y, pd.Series(True, index=y.index))["rmse"] == (
        pytest.approx(0.5)
    )

    y_nan = y.copy()
    y_nan[3] = np.nan
    s = em.score(FirstColumn(), X, y_nan, pd.Series(True, index=y.index))
    assert s["n"] == 3 and s["rmse"] == pytest.approx(0.0)


def test_clusters_fit_on_train_only(em):
    """UT-18 (NFR-1, FR-12)"""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    is_train = pd.Series(np.arange(200) < 150)

    before = em.fit_clusters(emb, is_train, k=4)
    changed = emb.copy()
    changed[~is_train.to_numpy()] = 50.0
    after = em.fit_clusters(changed, is_train, k=4)

    np.testing.assert_allclose(before.cluster_centers_, after.cluster_centers_)
    labels = before.predict(emb)
    assert labels.shape == (200,) and set(labels) <= set(range(4))
