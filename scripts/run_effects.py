"""OLS effects of the 14 rule features with channel-clustered CIs (FR-7).

Usage: uv run python scripts/run_effects.py
Writes reports/effects.csv and reports/usage_rates.csv.
"""

from __future__ import annotations

import logging

import pandas as pd

from ytclickbait.config import load_config
from ytclickbait.models.effects import effects_table, usage_rates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    videos = pd.read_parquet(cfg.file("videos"))
    rules = pd.read_parquet(cfg.file("rule_features"))

    effects = effects_table(videos, rules)
    out = cfg.root / "reports"
    out.mkdir(exist_ok=True)
    effects.to_csv(out / "effects.csv", index=False)
    usage_rates(rules).rename_axis("feature").reset_index().to_csv(
        out / "usage_rates.csv", index=False
    )
    overall = effects[(effects.scope == "overall") & (effects.target == "log_views")]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(overall[["model", "feature", "coef", "ci_low", "ci_high", "std_coef"]])


if __name__ == "__main__":
    main()
