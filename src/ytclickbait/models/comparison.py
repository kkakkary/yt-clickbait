"""Compare title-feature models on held-out data (FR-6, FT-3).

Every model is Ridge on [title features + the same controls]; only the title block changes.
The controls-only model is the floor each title model must beat. All fitting (scaling, alpha
search, coefficients) uses train rows only (NFR-1).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from ytclickbait.config import Config
from ytclickbait.features.tfidf import load_tfidf
from ytclickbait.models.effects import RULE_FEATURES
from ytclickbait.models.tfidf_model import (
    SPLITS,
    TARGETS,
    build_controls,
    fit_ridge,
    score,
    split_masks,
)

log = logging.getLogger(__name__)

# model name -> title blocks it uses (controls are always included)
MODELS = {
    "controls only": [],
    "rules": ["rules"],
    "rules + sentiment": ["rules", "sentiment"],
    "tfidf": ["tfidf"],
    "embeddings": ["embeddings"],
    "rules + embeddings": ["rules", "embeddings"],
}
PRIMARY_SPLIT = "split_group"  # new creators: the title can't borrow the creator's audience


def scale_on_train(block: np.ndarray, is_train: pd.Series) -> np.ndarray:
    """Standardize columns using the mean and spread of train rows only."""
    scaler = StandardScaler().fit(block[is_train.to_numpy()])
    return scaler.transform(block)


def assemble(controls: sparse.csr_matrix, blocks: list) -> sparse.csr_matrix | np.ndarray:
    """Join controls with title blocks; stay dense unless TF-IDF (sparse) is included."""
    if any(sparse.issparse(b) for b in blocks):
        return sparse.hstack([controls, *[sparse.csr_matrix(b) for b in blocks]]).tocsr()
    return np.hstack([controls.toarray(), *blocks])


def compare(
    videos: pd.DataFrame,
    split: pd.DataFrame,
    title_blocks: dict[str, np.ndarray | sparse.csr_matrix],
    models: dict[str, list[str]] = MODELS,
    targets: list[str] = TARGETS,
    splits: list[str] = SPLITS,
    seed: int = 42,
) -> pd.DataFrame:
    """RMSE / R² on test rows for every model × target × split, with gains over controls."""
    if "controls only" not in models:
        raise ValueError('models must include "controls only", the baseline for every gain')
    rows = []
    for split_name in splits:
        controls, _ = build_controls(videos, split, split_name)
        train, test = split_masks(split, split_name)
        groups = videos["channel_id"] if split_name == "split_group" else None
        scaled = {
            name: (b if sparse.issparse(b) else scale_on_train(np.asarray(b, dtype=float), train))
            for name, b in title_blocks.items()
            if name in {n for used in models.values() for n in used}
        }
        for model_name, used in models.items():
            X = assemble(controls, [scaled[n] for n in used])
            for target in targets:
                y = videos[target]
                log.info("%s | %s | %s", split_name, target, model_name)
                search = fit_ridge(X, y, train, groups=groups, seed=seed)
                res = score(search, X, y, test)
                rows.append(
                    {
                        "target": target,
                        "split": split_name,
                        "model": model_name,
                        "alpha": search.best_params_["alpha"],
                        "n_train": int((train.to_numpy() & y.notna().to_numpy()).sum()),
                        "n_test": res["n"],
                        "rmse": res["rmse"],
                        "r2": res["r2"],
                    }
                )
    out = pd.DataFrame(rows)
    base = out[out["model"] == "controls only"].set_index(["target", "split"])
    key = list(zip(out["target"], out["split"], strict=True))
    out["rmse_gain_vs_controls"] = base.loc[key, "rmse"].to_numpy() - out["rmse"]
    out["r2_gain_vs_controls"] = out["r2"] - base.loc[key, "r2"].to_numpy()
    out["primary"] = out["split"] == PRIMARY_SPLIT
    return out


def load_title_blocks(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read videos, split and every title-feature table; check all are row-aligned."""
    videos = pd.read_parquet(cfg.file("videos"))
    split = pd.read_parquet(cfg.file("split"))
    rules = pd.read_parquet(cfg.file("rule_features"))
    sentiment = pd.read_parquet(cfg.file("sentiment"))
    for name, table in [("split", split), ("rule_features", rules), ("sentiment", sentiment)]:
        if not videos["video_id"].equals(table["video_id"]):
            raise ValueError(f"{name} is out of sync with videos.parquet; rebuild the features")
    _, tfidf = load_tfidf(cfg.file("tfidf_matrix"), cfg.file("tfidf_vectorizer", "models"))
    embeddings = np.load(cfg.file("embeddings"))
    for name, n in [("tfidf", tfidf.shape[0]), ("embeddings", embeddings.shape[0])]:
        if n != len(videos):
            raise ValueError(f"{name} has {n} rows but videos has {len(videos)}")
    blocks = {
        "rules": rules[RULE_FEATURES].to_numpy(dtype=float),
        "sentiment": sentiment.drop(columns="video_id").to_numpy(dtype=float),
        "tfidf": tfidf,
        "embeddings": embeddings.astype(float),
    }
    return videos, split, blocks


def run(cfg: Config) -> pd.DataFrame:
    videos, split, blocks = load_title_blocks(cfg)
    return compare(videos, split, blocks, seed=cfg["seed"])
