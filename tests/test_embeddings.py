"""Embedding behaviour the spec relies on (FR-10, NFR-2, EA-1 - EA-3, EA-7, EA-8).

Most tests here run the real MiniLM model, so they are marked slow. They pin down what the
embeddings can and can't see; if one fails, update the matching EA row in docs/SPEC.md.
"""

import numpy as np
import pandas as pd
import pytest

from ytclickbait.features import embeddings
from ytclickbait.features.embeddings import embed_titles

TITLES = [
    "YOU WON'T BELIEVE THIS",  # 0
    "you won't believe this",  # 1
    "pasta night 😱",  # 2
    "pasta night 🔥",  # 3
    "great video!",  # 4
    "great video!!!",  # 5
    "car crash",  # 6
    "Official Trailer",  # 7
    "Official Trailer",  # 8
    "10 tips for cooking at home",  # 9
]


@pytest.fixture(scope="module")
def vecs():
    return embed_titles(pd.Series(TITLES), device="cpu")


@pytest.mark.slow
def test_vectors_are_unit_length_float32(vecs):
    """UT-8 (FR-10, EA-7)"""
    assert vecs.shape == (len(TITLES), 384)
    assert vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


@pytest.mark.slow
def test_caps_are_invisible(vecs):
    """UT-9 (EA-1): the model lowercases, so caps can't be studied with embeddings."""
    np.testing.assert_allclose(vecs[0], vecs[1], atol=1e-5)


@pytest.mark.slow
def test_different_emojis_look_alike(vecs):
    """UT-10 (EA-2): most emojis become the same unknown token."""
    assert vecs[2] @ vecs[3] > 0.99


@pytest.mark.slow
def test_exclamation_count_barely_matters(vecs):
    """UT-11 (EA-3)"""
    one_vs_three = vecs[4] @ vecs[5]
    assert one_vs_three > 0.9
    assert one_vs_three > vecs[4] @ vecs[6]


@pytest.mark.slow
def test_identical_titles_get_identical_vectors(vecs):
    """UT-12 (FR-10, EA-7)"""
    np.testing.assert_allclose(vecs[7], vecs[8], atol=1e-6)


def test_cache_recomputes_when_model_changes(tmp_path, monkeypatch):
    """UT-13 (FR-10, EA-8): the saved vectors are tied to the model name."""
    calls = []

    def fake_embed(titles, model_name, **kwargs):
        calls.append(model_name)
        return np.zeros((len(titles), 2), dtype=np.float32)

    monkeypatch.setattr(embeddings, "embed_titles", fake_embed)
    cache = tmp_path / "e.npy"
    titles = pd.Series(["one", "two"])
    embeddings.load_or_embed(titles, cache, model_name="model-a")
    embeddings.load_or_embed(titles, cache, model_name="model-a")
    assert calls == ["model-a"]
    embeddings.load_or_embed(titles, cache, model_name="model-b")
    assert calls == ["model-a", "model-b"]
    stamp = cache.with_suffix(".sha256").read_text()
    assert stamp == embeddings.titles_fingerprint(titles, "model-b")


@pytest.mark.slow
def test_cpu_and_accelerator_agree():
    """UT-14 (NFR-2): GPU results may differ from CPU, but only slightly."""
    from ytclickbait.features.sentiment import best_device

    device = best_device()
    if device == "cpu":
        pytest.skip("no GPU on this machine")
    titles = pd.Series(TITLES[:5])
    cpu = embed_titles(titles, device="cpu")
    gpu = embed_titles(titles, device=device)
    np.testing.assert_allclose(cpu, gpu, atol=1e-4)
