"""Fit FR-12 title clusters and write reports/embedding_clusters.csv.

Usage: uv run python scripts/run_embedding_clusters.py
"""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ytclickbait.config import load_config
from ytclickbait.features.embeddings import titles_fingerprint
from ytclickbait.models.embedding_model import fit_clusters
from ytclickbait.models.tfidf_model import build_controls, fit_ridge, split_masks


def main() -> None:
    cfg = load_config()
    videos = pd.read_parquet(cfg.file("videos"))
    split = pd.read_parquet(cfg.file("split"))
    if not videos["video_id"].equals(split["video_id"]):
        raise ValueError("videos.parquet and split.parquet are out of sync")

    embedding_path = cfg.file("embeddings")
    fingerprint = titles_fingerprint(videos["title"], cfg["embeddings"]["model"])
    if embedding_path.with_suffix(".sha256").read_text() != fingerprint:
        raise ValueError("Embeddings are out of sync with the titles or model")
    embeddings = np.load(embedding_path)
    if embeddings.shape != (len(videos), 384):
        raise ValueError("Expected one 384-dimensional embedding per video")

    split_name = "split_group"
    train, test = split_masks(split, split_name)
    target = videos["log_views"]
    scored = (test & target.notna()).to_numpy()
    controls, _ = build_controls(videos, split, split_name)

    with threadpool_limits(limits=1):
        # k = 8: clusters are seed-stable only up to k = 9, and 8 is the first k that separates
        # sports game highlights from sports talk. Evidence: scripts/run_k_sweep.py.
        kmeans = fit_clusters(embeddings, train, k=8, seed=cfg["seed"])
        labels = kmeans.predict(embeddings)
        baseline = fit_ridge(
            controls, target, train, groups=videos["channel_id"], seed=cfg["seed"]
        )
        residuals = target.to_numpy()[scored] - baseline.predict(controls[scored])

    rows = []
    for cluster, center in enumerate(kmeans.cluster_centers_):
        members = np.flatnonzero(labels == cluster)
        distances = np.linalg.norm(embeddings[members] - center, axis=1)
        closest = members[np.argsort(distances, kind="stable")[:5]]
        cluster_residuals = residuals[labels[scored] == cluster]
        row = {
            "cluster": cluster + 1,
            "n_videos": len(members),
            "n_test_scored": len(cluster_residuals),
            "mean_test_log_views_residual": (
                cluster_residuals.mean() if len(cluster_residuals) else np.nan
            ),
        }
        for rank, index in enumerate(closest, start=1):
            row[f"title_{rank}"] = videos["title"].iat[index]
        rows.append(row)

    report = cfg.root / "reports" / "embedding_clusters.csv"
    report.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(report, index=False, float_format="%.8f")
    print(f"Wrote {len(rows)} clusters to {report} (controls-only alpha={baseline.best_params_['alpha']})")


if __name__ == "__main__":
    main()
