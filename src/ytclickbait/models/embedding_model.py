"""Models on title embeddings (FR-11, FR-12). See docs/SPEC.md §5 for the interface."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_ALPHAS = tuple(np.logspace(-2, 4, 13))  # 0.01 ... 10,000


def fit_ridge(
    X, y: pd.Series, is_train: pd.Series, alphas=DEFAULT_ALPHAS, seed: int = 42
) -> Pipeline:
    """Scale the columns, then fit Ridge, using only train rows with a known target (NFR-1).

    The penalty strength (alpha) is picked by 5-fold cross-validation inside those rows, so the
    test rows never influence the scaling, the alpha choice or the coefficients.
    """
    y = np.asarray(y, dtype=float)
    keep = np.asarray(is_train, dtype=bool) & ~np.isnan(y)
    if keep.sum() < 5:
        raise ValueError(f"need at least 5 train rows with a known target, got {keep.sum()}")

    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=folds))
    return model.fit(np.asarray(X)[keep], y[keep])


def score(model, X, y: pd.Series, mask: pd.Series) -> dict:
    """RMSE, R² and row count of `model` on the rows where `mask` is True and `y` is known.

    Rows with a missing target (likes or comments turned off) are skipped, so `n` says how
    many rows were actually scored.
    """
    y = np.asarray(y, dtype=float)
    keep = np.asarray(mask, dtype=bool) & ~np.isnan(y)
    if not keep.any():
        raise ValueError("no rows to score: mask is empty or every target is missing")

    y_true = y[keep]
    y_pred = np.asarray(model.predict(np.asarray(X)[keep]), dtype=float)

    errors = y_true - y_pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    spread = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum(errors**2) / spread) if spread > 0 else float("nan")
    return {"rmse": rmse, "r2": r2, "n": int(keep.sum())}


def fit_clusters(emb, is_train: pd.Series, k: int, seed: int = 42) -> KMeans:
    """Group titles into `k` clusters with k-means, learned from train rows only (FR-12, NFR-1).

    Use `.predict(emb)` on the result to label every row, test rows included. The vectors are
    unit length, so the nearest centre is also the most similar in cosine terms.
    """
    train = np.asarray(emb)[np.asarray(is_train, dtype=bool)]
    if len(train) < k:
        raise ValueError(f"need at least k={k} train rows to make {k} clusters, got {len(train)}")
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(train)
