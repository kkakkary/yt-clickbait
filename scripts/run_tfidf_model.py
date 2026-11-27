"""Fit controls-only and TF-IDF Ridge models; write metrics and top terms to reports/.

Usage: uv run python scripts/run_tfidf_model.py
"""

from __future__ import annotations

import logging

import pandas as pd

from ytclickbait.config import load_config
from ytclickbait.models.tfidf_model import run

log = logging.getLogger("run_tfidf_model")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    metrics, terms = run(cfg)

    out = cfg.root / "reports"
    out.mkdir(exist_ok=True)
    metrics.to_csv(out / "tfidf_metrics.csv", index=False)
    terms.to_csv(out / "tfidf_top_terms.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(metrics.to_string(index=False))
    log.info("Wrote %s and %s", out / "tfidf_metrics.csv", out / "tfidf_top_terms.csv")


if __name__ == "__main__":
    main()
