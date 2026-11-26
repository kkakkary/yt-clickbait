"""Ridge on TF-IDF + controls for log_views and engagement (FR-6, NFR-1).

Everything learned from data (scaler, one-hot categories, Ridge, alpha) uses train rows only.
Rows with a missing target (likes/comments disabled) are skipped, never treated as zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ytclickbait.config import Config
from ytclickbait.features.tfidf import load_tfidf

TARGETS = ["log_views", "log_likes_per_view", "log_comments_per_view"]
SPLITS = ["split_group", "split_time"]
NUMERIC_CONTROLS = ["log_age_days", "n_tags", "log_desc_len", "n_countries"]
CATEGORICAL_CONTROLS = ["category_name", "publish_week", "publish_hour_utc", "publish_dow_utc"]
DEFAULT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def split_masks(split: pd.DataFrame, split_name: str) -> tuple[pd.Series, pd.Series]:
    """Boolean (train, test) masks for one split column."""
    return split[split_name] == "train", split[split_name] == "test"


def build_controls(
    videos: pd.DataFrame, split: pd.DataFrame, split_name: str
) -> tuple[sparse.csr_matrix, ColumnTransformer]:
    """Standardized numeric + one-hot controls, fit on the train rows of `split_name`.

    The channel baseline is added only for `split_time`: under `split_group` test channels
    never appear in train, so it carries no information there.
    """
    df = videos[["log_age_days", "n_tags", "desc_len", "n_countries"]].copy()
    df["log_desc_len"] = np.log1p(videos["desc_len"])
    df = df.drop(columns="desc_len")
    for col in CATEGORICAL_CONTROLS:
        df[col] = videos[col].astype(str)

    numeric = list(NUMERIC_CONTROLS)
    if split_name == "split_time":
        df["channel_baseline"] = split["channel_baseline_split_time"]
        # Channels outside the within-channel split have no baseline; they are never train/test.
        df["channel_baseline"] = df["channel_baseline"].fillna(df["channel_baseline"].mean())
        numeric.append("channel_baseline")

    encoder = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_CONTROLS),
        ],
        sparse_threshold=1.0,
    )
    train, _ = split_masks(split, split_name)
    encoder.fit(df[train.to_numpy()])
    return sparse.csr_matrix(encoder.transform(df)), encoder


def fit_ridge(
    X,
    y: pd.Series,
    is_train: pd.Series,
    groups: pd.Series | None = None,
    alphas=DEFAULT_ALPHAS,
    seed: int = 42,
    n_splits: int = 5,
) -> GridSearchCV:
    """Ridge with alpha picked by CV, using train rows with a non-missing target only.

    With `groups` (channel ids) CV folds keep each channel in one fold, which mimics the
    unseen-creator test of `split_group`.
    """
    use = (is_train.to_numpy()) & y.notna().to_numpy()
    X_fit, y_fit = X[use], y.to_numpy()[use]
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        fit_kwargs = {"groups": groups.to_numpy()[use]}
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit_kwargs = {}
    search = GridSearchCV(
        Ridge(random_state=seed),
        {"alpha": list(alphas)},
        cv=cv,
        scoring="neg_root_mean_squared_error",
    )
    search.fit(X_fit, y_fit, **fit_kwargs)
    return search


def score(model, X, y: pd.Series, mask: pd.Series) -> dict:
    """RMSE, R² and row count on rows where `mask` is true and `y` is not missing."""
    use = mask.to_numpy() & y.notna().to_numpy()
    truth = y.to_numpy()[use]
    pred = model.predict(X[use])
    return {
        "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
        "r2": float(r2_score(truth, pred)),
        "n": int(use.sum()),
    }


def top_terms(coef: np.ndarray, vocab: np.ndarray, n_controls: int, n: int = 30) -> pd.DataFrame:
    """Most positive and most negative TF-IDF coefficients (control columns excluded)."""
    term_coef = np.asarray(coef)[n_controls:]
    order = term_coef.argsort()
    rows = [("positive", i) for i in order[::-1][:n]] + [("negative", i) for i in order[:n]]
    return pd.DataFrame(
        {
            "direction": [d for d, _ in rows],
            "term": [vocab[i] for _, i in rows],
            "coef": [term_coef[i] for _, i in rows],
        }
    )


def run(cfg: Config, targets=TARGETS, splits=SPLITS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Controls-only vs TF-IDF + controls for every target and split."""
    videos = pd.read_parquet(cfg.file("videos"))
    split = pd.read_parquet(cfg.file("split"))
    if not videos["video_id"].equals(split["video_id"]):
        raise ValueError("videos.parquet and split.parquet are out of sync; rerun build_dataset.py")
    vectorizer, tfidf = load_tfidf(cfg.file("tfidf_matrix"), cfg.file("tfidf_vectorizer", "models"))
    if tfidf.shape[0] != len(videos):
        raise ValueError("tfidf.npz is out of sync with videos.parquet; rerun build_features.py")
    vocab = vectorizer.get_feature_names_out()
    seed = cfg["seed"]

    metrics, terms = [], []
    for split_name in splits:
        controls, _ = build_controls(videos, split, split_name)
        n_controls = controls.shape[1]
        features = {
            "controls": controls,
            "tfidf+controls": sparse.hstack([controls, tfidf]).tocsr(),
        }
        train, test = split_masks(split, split_name)
        groups = videos["channel_id"] if split_name == "split_group" else None
        for target in targets:
            y = videos[target]
            base_r2 = None
            for model_name, X in features.items():
                search = fit_ridge(X, y, train, groups=groups, seed=seed)
                res = score(search, X, y, test)
                if model_name == "controls":
                    base_r2 = res["r2"]
                metrics.append(
                    {
                        "target": target,
                        "split": split_name,
                        "model": model_name,
                        "alpha": search.best_params_["alpha"],
                        "n_train": int((train.to_numpy() & y.notna().to_numpy()).sum()),
                        "n_test": res["n"],
                        "rmse": res["rmse"],
                        "r2": res["r2"],
                        "r2_gain_vs_controls": res["r2"] - base_r2,
                    }
                )
                if model_name == "tfidf+controls":
                    t = top_terms(search.best_estimator_.coef_, vocab, n_controls)
                    t.insert(0, "split", split_name)
                    t.insert(0, "target", target)
                    terms.append(t)
    return pd.DataFrame(metrics), pd.concat(terms, ignore_index=True)
