"""Models on title embeddings (FR-11, FR-12). See docs/SPEC.md §5 for the interface.

`fit_ridge` and `score` are the shared versions from `tfidf_model`, so every Ridge model in the
project (controls, rules, TF-IDF, embeddings) is fitted and graded the same way, and UT-15 –
UT-17 test the code the pipeline actually runs.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ytclickbait.models.tfidf_model import fit_ridge, score

__all__ = ["fit_clusters", "fit_ridge", "score"]


def fit_clusters(emb, is_train: pd.Series, k: int, seed: int = 42) -> KMeans:
    """Group titles into `k` clusters with k-means, learned from train rows only (FR-12, NFR-1).

    Use `.predict(emb)` on the result to label every row, test rows included. The vectors are
    unit length, so the nearest centre is also the most similar in cosine terms.
    """
    train = np.asarray(emb)[np.asarray(is_train, dtype=bool)]
    if len(train) < k:
        raise ValueError(f"need at least k={k} train rows to make {k} clusters, got {len(train)}")
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(train)
