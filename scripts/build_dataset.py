"""Raw Kaggle files -> data/processed/videos.parquet and split.parquet.

Usage: uv run python scripts/build_dataset.py
"""

from __future__ import annotations

import logging

import pandas as pd

from ytclickbait.config import load_config
from ytclickbait.data.clean import clean
from ytclickbait.data.language import add_language_flags
from ytclickbait.data.load import load_all
from ytclickbait.data.split import channel_baseline, group_split, within_channel_time_split

log = logging.getLogger("build_dataset")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config()
    out_dir = cfg.path("processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, categories = load_all(cfg.path("raw"), cfg["countries"])
    log.info(
        "Snapshot rows: %d, date range %s to %s",
        len(raw),
        raw["trending_date"].min().date(),
        raw["trending_date"].max().date(),
    )

    videos = clean(raw, categories)
    lang = cfg["language"]
    videos = add_language_flags(videos, lang["min_letters"], lang["candidates"])

    dropped = videos.loc[~videos["is_english"]]
    by_country = dropped["countries"].explode().value_counts()
    log.info(
        "Non-English videos dropped: %d of %d (by country they trended in: %s)",
        len(dropped),
        len(videos),
        by_country.to_dict(),
    )
    videos = videos[videos["is_english"]].reset_index(drop=True)
    # Recount after dropping so it reflects the analysed set.
    videos["channel_n_videos"] = videos.groupby("channel_id")["video_id"].transform("size")

    log.info(
        "Videos with more than one title: %d (%.1f%%)",
        videos["title_changed"].sum(),
        100 * videos["title_changed"].mean(),
    )

    split_cfg = cfg["split"]
    split = pd.DataFrame(
        {
            "video_id": videos["video_id"],
            "split_group": group_split(videos, split_cfg["test_size"], cfg["seed"]),
            "split_time": within_channel_time_split(
                videos, split_cfg["min_channel_videos"], split_cfg["test_size"]
            ),
        }
    )
    # Channel baselines depend on which rows are train, so one per split.
    for name in ["split_group", "split_time"]:
        is_train = split[name] == "train"
        split[f"channel_baseline_{name}"] = channel_baseline(videos, is_train)
    log.info(
        "Split counts:\n%s", split[["split_group", "split_time"]].apply(pd.Series.value_counts)
    )

    videos.to_parquet(cfg.file("videos"), index=False)
    split.to_parquet(cfg.file("split"), index=False)
    log.info("Wrote %d videos to %s", len(videos), cfg.file("videos"))


if __name__ == "__main__":
    main()
