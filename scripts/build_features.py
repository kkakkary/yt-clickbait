"""videos.parquet -> rule features, sentiment, TF-IDF, embeddings (all row-aligned by video_id).

Usage: uv run python scripts/build_features.py [--skip-roberta] [--skip-embeddings]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from ytclickbait.config import load_config
from ytclickbait.features.embeddings import load_or_embed
from ytclickbait.features.rules import rule_features
from ytclickbait.features.sentiment import sentiment_features
from ytclickbait.features.tfidf import fit_tfidf, save_tfidf, top_terms

log = logging.getLogger("build_features")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-roberta", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()

    videos = pd.read_parquet(cfg.file("videos"), columns=["video_id", "title"])
    split = pd.read_parquet(cfg.file("split"))
    if not videos["video_id"].equals(split["video_id"]):
        raise ValueError("videos.parquet and split.parquet are out of sync; rerun build_dataset.py")
    titles = videos["title"]

    rules = rule_features(titles)
    rules.insert(0, "video_id", videos["video_id"])
    rules.to_parquet(cfg.file("rule_features"), index=False)
    log.info("Rule features: %s", rules.shape)

    s = cfg["sentiment"]
    sent = sentiment_features(
        titles,
        use_roberta=not args.skip_roberta,
        model_name=s["roberta_model"],
        batch_size=s["batch_size"],
    )
    sent.insert(0, "video_id", videos["video_id"])
    sent.to_parquet(cfg.file("sentiment"), index=False)
    log.info("Sentiment features: %s", sent.shape)

    t = cfg["tfidf"]
    # Fit vocabulary only on rows that are train under both splits, so neither split's
    # test titles shape the vocabulary or IDF weights.
    is_train = (split["split_group"] == "train") & (split["split_time"] != "test")
    vectorizer, matrix = fit_tfidf(
        titles, is_train, tuple(t["ngram_range"]), t["min_df"], t["max_df"], t["sublinear_tf"]
    )
    save_tfidf(vectorizer, matrix, cfg.file("tfidf_matrix"), cfg.file("tfidf_vectorizer", "models"))
    log.info(
        "TF-IDF: %s, top terms: %s", matrix.shape, list(top_terms(vectorizer, matrix, 15).index)
    )

    if not args.skip_embeddings:
        e = cfg["embeddings"]
        vectors = load_or_embed(
            titles,
            cfg.file("embeddings"),
            model_name=e["model"],
            batch_size=e["batch_size"],
        )
        log.info("Embeddings: %s", vectors.shape)


if __name__ == "__main__":
    main()
