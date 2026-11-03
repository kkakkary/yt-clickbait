import numpy as np
import pandas as pd
import pytest

from ytclickbait.features import embeddings
from ytclickbait.features.embeddings import embed_titles
from ytclickbait.features.tfidf import fit_tfidf, load_tfidf, save_tfidf, top_terms

TITLES = pd.Series(["$1 vs $1,000 hotel", "5 tips for cooking", "cooking tips", "unseenword here"])
IS_TRAIN = pd.Series([True, True, True, False])


def test_fit_on_train_only():
    vec, X = fit_tfidf(TITLES, IS_TRAIN, min_df=1, max_df=1.0)
    vocab = set(vec.get_feature_names_out())
    assert "unseenword" not in vocab
    assert X.shape[0] == len(TITLES)
    assert X[3].nnz == 0


def test_keeps_numbers_and_dollars():
    vec, _ = fit_tfidf(TITLES, IS_TRAIN, min_df=1, max_df=1.0)
    vocab = set(vec.get_feature_names_out())
    assert {"$1", "5", "cooking tips"} <= vocab


def test_round_trip(tmp_path):
    vec, X = fit_tfidf(TITLES, IS_TRAIN, min_df=1, max_df=1.0)
    save_tfidf(vec, X, tmp_path / "m.npz", tmp_path / "v.pkl")
    vec2, X2 = load_tfidf(tmp_path / "m.npz", tmp_path / "v.pkl")
    assert (X != X2).nnz == 0
    assert list(vec2.get_feature_names_out()) == list(vec.get_feature_names_out())
    assert len(top_terms(vec, X, n=3)) == 3


def test_embedding_cache_rejects_changed_titles(tmp_path, monkeypatch):
    calls = []

    def fake_embed(titles, **kwargs):
        calls.append(list(titles))
        return np.full((len(titles), 2), len(calls), dtype=np.float32)

    monkeypatch.setattr(embeddings, "embed_titles", fake_embed)
    cache = tmp_path / "e.npy"
    a = pd.Series(["one", "two"])
    embeddings.load_or_embed(a, cache)
    embeddings.load_or_embed(a, cache)  # same titles: cache hit
    assert len(calls) == 1
    out = embeddings.load_or_embed(pd.Series(["two", "one"]), cache)  # same count, new order
    assert len(calls) == 2 and out[0, 0] == 2


@pytest.mark.slow
def test_embeddings_shape_and_similarity():
    v = embed_titles(
        pd.Series(["cooking pasta at home", "making pasta in my kitchen", "car crash"])
    )
    assert v.shape == (3, 384) and v.dtype == np.float32
    assert v[0] @ v[1] > v[0] @ v[2]
