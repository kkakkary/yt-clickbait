"""Compare k-means cluster counts for FR-12 and write reports/k_sweep.csv.

Same setup as scripts/run_embedding_clusters.py (split_group, train rows only, controls-only
Ridge residuals on held-out rows), for k = 4 to 16. One row per k:

- inertia: total squared distance to the nearest centre (lower = tighter; always falls with k)
- silhouette: separation of clusters on a 10,000-row train sample (-1 to 1, higher = cleaner)
- stability_ari: agreement between labels from seed and seed + 1 (1 = identical clusters)
- min_test_per_cluster: held-out videos in the smallest cluster (residual reliability)
- max_top3_channel_share: largest share of one cluster's held-out videos from its top 3 channels
- n_clusters_top3_over_40pct: clusters whose held-out videos are > 40% from 3 channels
- residual_eta2: share of held-out residual variance explained by cluster membership
- residual_range: highest minus lowest cluster mean residual

Usage: uv run python scripts/run_k_sweep.py   (about 15 min on one CPU thread)
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from ytclickbait.config import load_config
from ytclickbait.features.embeddings import titles_fingerprint
from ytclickbait.models.embedding_model import fit_clusters
from ytclickbait.models.tfidf_model import build_controls, fit_ridge, split_masks

K_RANGE = range(4, 17)
SILHOUETTE_SAMPLE = 10_000  # silhouette cost grows with n^2, so score a train sample
CHANNEL_SHARE_FLAG = 0.4


def channel_top3_share(channels: np.ndarray) -> float:
    counts = pd.Series(channels).value_counts()
    return counts.head(3).sum() / counts.sum() if len(counts) else np.nan


def main() -> None:
    cfg = load_config()
    seed = cfg["seed"]
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
    channels = videos["channel_id"].to_numpy()[scored]
    rng = np.random.default_rng(seed)
    sample = rng.choice(np.flatnonzero(train.to_numpy()), SILHOUETTE_SAMPLE, replace=False)

    rows = []
    with threadpool_limits(limits=1):
        baseline = fit_ridge(controls, target, train, groups=videos["channel_id"], seed=seed)
        residuals = target.to_numpy()[scored] - baseline.predict(controls[scored])
        total_ss = ((residuals - residuals.mean()) ** 2).sum()

        for k in K_RANGE:
            start = time.time()
            kmeans = fit_clusters(embeddings, train, k=k, seed=seed)
            labels = kmeans.predict(embeddings)
            other = fit_clusters(embeddings, train, k=k, seed=seed + 1).predict(embeddings)

            test_labels = labels[scored]
            sizes = np.bincount(test_labels, minlength=k)
            means = np.array([residuals[test_labels == c].mean() for c in range(k)])
            shares = np.array([channel_top3_share(channels[test_labels == c]) for c in range(k)])
            rows.append(
                {
                    "k": k,
                    "inertia": kmeans.inertia_,
                    "silhouette": silhouette_score(
                        embeddings[sample], kmeans.predict(embeddings[sample])
                    ),
                    "stability_ari": adjusted_rand_score(labels, other),
                    "min_test_per_cluster": int(sizes.min()),
                    "max_top3_channel_share": float(np.nanmax(shares)),
                    "n_clusters_top3_over_40pct": int((shares > CHANNEL_SHARE_FLAG).sum()),
                    "residual_eta2": (sizes * (means - residuals.mean()) ** 2).sum() / total_ss,
                    "residual_range": float(means.max() - means.min()),
                }
            )
            print(f"k={k} done in {time.time() - start:.0f}s", flush=True)

    report = cfg.root / "reports" / "k_sweep.csv"
    report.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(report, index=False, float_format="%.8f")
    print(f"Wrote {len(rows)} rows to {report}")


if __name__ == "__main__":
    main()
