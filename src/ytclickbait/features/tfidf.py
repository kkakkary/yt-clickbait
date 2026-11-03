"""TF-IDF over titles. Fit on train rows only so test vocabulary doesn't leak in."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Default token pattern drops 1-char tokens; this keeps single digits like "5" and "$1".
TOKEN_PATTERN = r"(?u)\$?\b\w+\b"


def fit_tfidf(
    titles: pd.Series,
    is_train: pd.Series,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 5,
    max_df: float = 0.9,
    sublinear_tf: bool = True,
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=tuple(ngram_range),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    vectorizer.fit(titles[is_train.to_numpy()])
    return vectorizer, vectorizer.transform(titles).tocsr()


def top_terms(vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix, n: int = 30) -> pd.Series:
    """Terms with the highest mean TF-IDF weight, for sanity checks."""
    means = np.asarray(matrix.mean(axis=0)).ravel()
    vocab = vectorizer.get_feature_names_out()
    order = means.argsort()[::-1][:n]
    return pd.Series(means[order], index=vocab[order], name="mean_tfidf")


def save_tfidf(
    vectorizer: TfidfVectorizer, matrix: sparse.csr_matrix, matrix_path: Path, vec_path: Path
):
    sparse.save_npz(matrix_path, matrix)
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_tfidf(matrix_path: Path, vec_path: Path) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    # Only load pickles this project wrote itself.
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer, sparse.load_npz(matrix_path)
