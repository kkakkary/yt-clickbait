"""Time the embedding steps at growing data sizes (NFR-3 scaling) -> reports/embedding_timing.csv.

For the first 10k, 40k and all titles (videos.parquet order), on CPU:

- embed_seconds: embed_titles (MiniLM, batch 256), including building the model (~1 s)
- titles_per_second: titles / embed_seconds (flat across sizes = linear scaling)
- kmeans_seconds: fit_clusters (k = 8, n_init = 10) on those rows' split_group train rows
- file_mb: size of the vectors as saved (titles x 384 x 4 bytes)

It also times reloading the full cached embeddings.npy (the path later runs take).
Budget (NFR-3): all titles in < 10 min; cache reload < 5 s.

Usage: uv run python scripts/time_embeddings.py   (about 10-15 min)
"""

import platform
import time

import numpy as np
import pandas as pd

from ytclickbait.config import load_config
from ytclickbait.features.embeddings import embed_titles
from ytclickbait.models.embedding_model import fit_clusters
from ytclickbait.models.tfidf_model import split_masks

SIZES = [10_000, 40_000]
EMBED_BUDGET_S = 600
RELOAD_BUDGET_S = 5


def main() -> None:
    cfg = load_config()
    videos = pd.read_parquet(cfg.file("videos"), columns=["video_id", "title"])
    split = pd.read_parquet(cfg.file("split"))
    if not videos["video_id"].equals(split["video_id"]):
        raise ValueError("videos.parquet and split.parquet are out of sync")
    train, _ = split_masks(split, "split_group")
    model = cfg["embeddings"]["model"]

    embed_titles(videos["title"].head(100), model, device="cpu")  # download / warm-up, untimed

    rows = []
    for n in [*SIZES, len(videos)]:
        titles = videos["title"].head(n)
        start = time.perf_counter()
        vecs = embed_titles(titles, model, device="cpu")
        embed_s = time.perf_counter() - start

        start = time.perf_counter()
        fit_clusters(vecs, train.head(n), k=8, seed=cfg["seed"])
        kmeans_s = time.perf_counter() - start

        rows.append(
            {
                "n_titles": n,
                "embed_seconds": embed_s,
                "titles_per_second": n / embed_s,
                "kmeans_seconds": kmeans_s,
                "file_mb": vecs.nbytes / 1e6,
            }
        )
        print(f"{n:>6} titles: embed {embed_s:6.1f} s, k-means {kmeans_s:5.1f} s", flush=True)

    start = time.perf_counter()
    np.load(cfg.file("embeddings"))
    reload_s = time.perf_counter() - start

    report = pd.DataFrame(rows)
    report["machine"] = f"{platform.machine()} {platform.platform(terse=True)}, CPU"
    report["full_reload_seconds"] = reload_s
    path = cfg.root / "reports" / "embedding_timing.csv"
    report.to_csv(path, index=False, float_format="%.3f")

    full = rows[-1]["embed_seconds"]
    print(
        f"Full embed {full:.0f} s vs budget {EMBED_BUDGET_S} s: "
        f"{'met' if full < EMBED_BUDGET_S else 'NOT met'}"
    )
    print(
        f"Cache reload {reload_s:.2f} s vs budget {RELOAD_BUDGET_S} s: "
        f"{'met' if reload_s < RELOAD_BUDGET_S else 'NOT met'}"
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
