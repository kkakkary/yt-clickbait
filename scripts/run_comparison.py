"""Compare controls-only, rules, TF-IDF and embedding Ridge models (FT-3).

Usage: uv run python scripts/run_comparison.py
Writes reports/model_comparison.csv.
"""

from __future__ import annotations

import logging

import pandas as pd

from ytclickbait.config import load_config
from ytclickbait.models.comparison import run


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    table = run(cfg)
    out = cfg.root / "reports"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "model_comparison.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.4f}".format):
        print(table.drop(columns="primary").to_string(index=False))


if __name__ == "__main__":
    main()
