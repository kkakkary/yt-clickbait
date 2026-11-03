"""Sentence embeddings of titles (all-MiniLM-L6-v2, 384 dims). Rows follow the input order."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ytclickbait.features.sentiment import best_device


def embed_titles(
    titles: pd.Series,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device or best_device())
    vectors = model.encode(
        titles.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def titles_fingerprint(titles: pd.Series, model_name: str) -> str:
    h = hashlib.sha256(model_name.encode())
    for t in titles:
        h.update(t.encode("utf-8") + b"\0")
    return h.hexdigest()


def load_or_embed(
    titles: pd.Series,
    cache: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    **kwargs,
) -> np.ndarray:
    """Reuse cached embeddings only if the same titles, in the same order, with the same
    model produced them. The fingerprint sits next to the cache file."""
    stamp = cache.with_suffix(".sha256")
    fingerprint = titles_fingerprint(titles, model_name)
    if cache.exists() and stamp.exists() and stamp.read_text() == fingerprint:
        return np.load(cache)
    vectors = embed_titles(titles, model_name=model_name, **kwargs)
    np.save(cache, vectors)
    stamp.write_text(fingerprint)
    return vectors
