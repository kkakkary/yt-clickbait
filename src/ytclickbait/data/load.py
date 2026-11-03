"""Read raw Kaggle files: {CC}_youtube_trending_data.csv and {CC}_category_id.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

MAX_BAD_DATE_SHARE = 0.01

RAW_COLUMNS = [
    "video_id",
    "title",
    "publishedAt",
    "channelId",
    "channelTitle",
    "categoryId",
    "trending_date",
    "tags",
    "view_count",
    "likes",
    "comment_count",
    "comments_disabled",
    "ratings_disabled",
    "description",
]


def load_country_csv(raw_dir: Path, country: str) -> pd.DataFrame:
    path = Path(raw_dir) / f"{country}_youtube_trending_data.csv"
    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        log.warning("%s is not UTF-8, falling back to latin-1", path.name)
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df = df[RAW_COLUMNS].copy()
    df["country"] = country
    for col in ["publishedAt", "trending_date"]:
        parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
        # An older Kaggle dataset used "yy.dd.mm" trending dates, which would all parse
        # to NaT here and then be silently dropped. Fail loudly instead.
        bad = parsed.isna() & df[col].notna()
        if bad.mean() > MAX_BAD_DATE_SHARE:
            raise ValueError(
                f"{path.name}: {bad.sum()} of {len(df)} {col} values didn't parse, "
                f"e.g. {df.loc[bad, col].iloc[0]!r}. Check the date format."
            )
        df[col] = parsed
    df["categoryId"] = pd.to_numeric(df["categoryId"], errors="coerce").astype("Int64")
    for col in ["view_count", "likes", "comment_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["comments_disabled", "ratings_disabled"]:
        df[col] = df[col].astype(str).str.lower().eq("true")
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    return df


def load_categories(raw_dir: Path, countries: list[str]) -> dict[int, str]:
    """Union of category id -> name across countries. Conflicting names are logged."""
    mapping: dict[int, str] = {}
    for country in countries:
        path = Path(raw_dir) / f"{country}_category_id.json"
        with open(path, encoding="utf-8") as f:
            items = json.load(f)["items"]
        for item in items:
            cid, name = int(item["id"]), item["snippet"]["title"]
            if cid in mapping and mapping[cid] != name:
                log.warning("Category %d: %r vs %r (%s)", cid, mapping[cid], name, country)
                continue
            mapping.setdefault(cid, name)
    return mapping


def load_all(raw_dir: Path, countries: list[str]) -> tuple[pd.DataFrame, dict[int, str]]:
    frames = []
    for country in countries:
        df = load_country_csv(raw_dir, country)
        log.info("%s: %d rows, %d unique videos", country, len(df), df["video_id"].nunique())
        frames.append(df)
    return pd.concat(frames, ignore_index=True), load_categories(raw_dir, countries)
