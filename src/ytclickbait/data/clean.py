"""Merge country files into one row per video, with targets and confounder controls.

Each video shows up once per day it trended, in every country it trended in. We keep
the latest snapshot (highest views) and summarise the history in extra columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _title_history(df: pd.DataFrame) -> pd.DataFrame:
    """Per video: how many distinct titles it had, and its first and last title."""
    ordered = df.sort_values(["trending_date", "view_count"])
    stripped = ordered["title"].str.strip()
    g = ordered.assign(_t=stripped).groupby("video_id")
    return pd.DataFrame(
        {
            "n_distinct_titles": g["_t"].nunique(),
            "first_title": g["_t"].first(),
            "last_title": g["_t"].last(),
        }
    )


def _trending_history(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("video_id")
    return pd.DataFrame(
        {
            "countries": g["country"].agg(lambda s: sorted(set(s))),
            "first_trending_date": g["trending_date"].min(),
            "days_trending": g["trending_date"].agg(lambda s: s.dt.normalize().nunique()),
        }
    )


def _count_tags(tags: pd.Series) -> pd.Series:
    tags = tags.fillna("[None]").astype(str).str.strip()
    counts = tags.str.split("|").str.len()
    return counts.where(~tags.isin(["[None]", ""]), 0).astype(int)


def clean(raw: pd.DataFrame, categories: dict[int, str]) -> pd.DataFrame:
    n0 = len(raw)
    df = raw.dropna(subset=["video_id", "channelId", "trending_date", "publishedAt", "view_count"])
    df = df[df["view_count"] > 0]
    log.info("Dropped %d snapshot rows with missing ids/dates or zero views", n0 - len(df))

    history = _title_history(df).join(_trending_history(df))

    # drop_duplicates, not groupby().last(): last() takes the last non-null value per
    # column, which can mix fields from different snapshots.
    latest = (
        df.sort_values(["trending_date", "view_count"])
        .drop_duplicates("video_id", keep="last")
        .set_index("video_id")
        .join(history)
        .reset_index()
    )
    out = latest.rename(
        columns={
            "channelId": "channel_id",
            "channelTitle": "channel_title",
            "categoryId": "category_id",
            "publishedAt": "published_at",
            "country": "snapshot_country",
        }
    )

    out["n_countries"] = out["countries"].str.len()
    out["title_changed"] = out["n_distinct_titles"] > 1

    out["category_name"] = out["category_id"].map(categories).fillna("Unknown")
    unknown = (out["category_name"] == "Unknown").sum()
    if unknown:
        log.warning("%d videos have a category id not found in the category files", unknown)

    # Controls
    age = (out["trending_date"] - out["published_at"]).dt.total_seconds() / 86400
    out["age_days"] = age.clip(lower=1)
    out["log_age_days"] = np.log1p(out["age_days"])
    # UTC, not the creator's local time: the data has no creator time zone.
    out["publish_hour_utc"] = out["published_at"].dt.hour
    out["publish_dow_utc"] = out["published_at"].dt.day_of_week
    out["publish_week"] = out["published_at"].dt.tz_localize(None).dt.to_period("W").dt.start_time
    out["n_tags"] = _count_tags(out["tags"])
    out["desc_len"] = out["description"].str.len()
    out["channel_n_videos"] = out.groupby("channel_id")["video_id"].transform("size")

    # Targets. Engagement is NaN (not 0) when the creator turned that signal off.
    out["log_views"] = np.log1p(out["view_count"])
    likes = out["likes"].where(~out["ratings_disabled"])
    comments = out["comment_count"].where(~out["comments_disabled"])
    out["likes_per_view"] = likes / out["view_count"]
    out["comments_per_view"] = comments / out["view_count"]
    out["log_likes_per_view"] = np.log1p(likes) - out["log_views"]
    out["log_comments_per_view"] = np.log1p(comments) - out["log_views"]

    log.info("%d snapshot rows -> %d unique videos", len(df), len(out))
    return out.sort_values("video_id").reset_index(drop=True)
