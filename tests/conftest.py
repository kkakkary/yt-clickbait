"""Tiny synthetic raw files shaped like the Kaggle dataset, so tests need no real data."""

from __future__ import annotations

import json

import pandas as pd
import pytest

ROWS = [
    # id, title, published, channel, category, trending, views, likes, comments,
    # comments_disabled, ratings_disabled, country
    (
        "v1",
        "YOU WON'T BELIEVE THIS!!!",
        "2021-01-01T10:00:00Z",
        "c1",
        24,
        "2021-01-02T00:00:00Z",
        1000,
        100,
        10,
        False,
        False,
        "US",
    ),
    (
        "v1",
        "YOU WON'T BELIEVE THIS!!!",
        "2021-01-01T10:00:00Z",
        "c1",
        24,
        "2021-01-03T00:00:00Z",
        5000,
        400,
        40,
        False,
        False,
        "US",
    ),
    (
        "v1",
        "You Won't Believe This",
        "2021-01-01T10:00:00Z",
        "c1",
        24,
        "2021-01-04T00:00:00Z",
        9000,
        700,
        70,
        False,
        False,
        "GB",
    ),
    (
        "v2",
        "Calm piano music for studying",
        "2021-02-01T08:00:00Z",
        "c2",
        10,
        "2021-02-03T00:00:00Z",
        2000,
        50,
        0,
        True,
        False,
        "CA",
    ),
    (
        "v3",
        "Je suis allé au marché ce matin avec ma famille",
        "2021-03-01T08:00:00Z",
        "c3",
        22,
        "2021-03-02T00:00:00Z",
        300,
        30,
        3,
        False,
        True,
        "CA",
    ),
    (
        "v4",
        "Zero views video",
        "2021-03-01T08:00:00Z",
        "c3",
        22,
        "2021-03-02T00:00:00Z",
        0,
        0,
        0,
        False,
        False,
        "US",
    ),
]


def _frame(rows):
    cols = [
        "video_id",
        "title",
        "publishedAt",
        "channelId",
        "categoryId",
        "trending_date",
        "view_count",
        "likes",
        "comment_count",
        "comments_disabled",
        "ratings_disabled",
        "country",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["channelTitle"] = df["channelId"].str.upper()
    df["tags"] = ["a|b|c", "a|b|c", "a|b|c", "[None]", "x", "[None]"][: len(df)]
    df["description"] = "desc"
    df["dislikes"] = 0
    return df


@pytest.fixture
def raw_dir(tmp_path):
    df = _frame(ROWS)
    cats = {
        "items": [
            {"id": "10", "snippet": {"title": "Music"}},
            {"id": "22", "snippet": {"title": "People & Blogs"}},
            {"id": "24", "snippet": {"title": "Entertainment"}},
        ]
    }
    for country in ["US", "GB", "CA"]:
        part = df[df["country"] == country].drop(columns="country")
        part.to_csv(tmp_path / f"{country}_youtube_trending_data.csv", index=False)
        (tmp_path / f"{country}_category_id.json").write_text(json.dumps(cats))
    return tmp_path
