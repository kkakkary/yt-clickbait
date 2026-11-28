"""Checks on the real processed tables: types, ranges, missing values and joins (UT-46).

These read data/processed/ (git-lfs), so they are marked full_data and skip when the files are
missing or still LFS pointers. CI runs them in the `data-checks` job after `git lfs pull`.
"""

import numpy as np
import pandas as pd
import pytest

from ytclickbait.config import load_config

pytestmark = pytest.mark.full_data

CFG = load_config()
TABLES = ["videos", "split", "rule_features", "sentiment"]


def real_file(name):
    path = CFG.file(name)
    if not path.exists() or path.stat().st_size <= 1024:
        pytest.skip(f"{path.name} not here (run build_dataset.py or git lfs pull)")
    return path


@pytest.fixture(scope="module")
def tables():
    return {name: pd.read_parquet(real_file(name)) for name in TABLES}


def test_all_tables_join_on_video_id(tables):
    """Joins: one row per video, same order in every table (NFR-4)."""
    videos = tables["videos"]
    assert 0.99 * 78_180 <= len(videos) <= 1.01 * 78_180
    assert videos["video_id"].notna().all() and videos["video_id"].is_unique
    for name in TABLES[1:]:
        assert tables[name]["video_id"].equals(videos["video_id"]), name


def test_video_types_and_ranges(tables):
    """Types and ranges: counts are whole and non-negative, dates are UTC, logs match counts."""
    v = tables["videos"]
    for col in ["view_count", "likes", "comment_count", "n_tags", "desc_len"]:
        assert pd.api.types.is_integer_dtype(v[col]), col
        assert (v[col] >= 0).all(), col
    assert str(v["published_at"].dt.tz) == "UTC"
    # trending_date is a day with no time of day, so a video published later that same day can
    # trend "before" it was published (55 videos, < 9 h). age_days is clipped to 1 for these.
    assert (v["trending_date"] >= v["published_at"] - pd.Timedelta(days=1)).all()
    assert (v["age_days"] >= 1).all()
    assert v["publish_hour_utc"].between(0, 23).all()
    assert v["publish_dow_utc"].between(0, 6).all()
    assert v["n_countries"].between(1, 3).all()
    assert set(v["snapshot_country"]) <= {"US", "GB", "CA"}
    np.testing.assert_allclose(v["log_views"], np.log(v["view_count"]), atol=1e-3)
    for col in ["likes_per_view", "comments_per_view"]:
        assert v[col].dropna().between(0, 1).all(), col


def test_missing_values_only_where_expected(tables):
    """Missing values: none in keys or titles; engagement is missing only when YouTube hides it."""
    v = tables["videos"]
    for col in ["video_id", "channel_id", "title", "published_at", "view_count", "log_views"]:
        assert v[col].notna().all(), col
    assert (v["title"].str.strip() != "").all()
    assert v["is_english"].all()
    assert v["likes_per_view"].isna().equals(v["ratings_disabled"])
    assert v["comments_per_view"].isna().equals(v["comments_disabled"])


def test_feature_tables_in_range(tables):
    """Rule and sentiment features stay in their defined ranges, with no gaps."""
    rules, sentiment = tables["rule_features"], tables["sentiment"]
    assert rules.drop(columns="video_id").notna().all().all()
    assert sentiment.drop(columns="video_id").notna().all().all()
    assert rules["caps_ratio"].between(0, 1).all()
    assert (rules.drop(columns="video_id") >= 0).all().all()
    for col in ["vader_neg", "vader_neu", "vader_pos", "roberta_negative", "roberta_positive"]:
        assert sentiment[col].between(0, 1).all(), col
    for col in ["vader_compound", "roberta_score"]:
        assert sentiment[col].between(-1, 1).all(), col


def test_split_labels_and_no_shared_channels(tables):
    """split_group keeps whole channels on one side (NFR-1); labels are the expected ones."""
    split, videos = tables["split"], tables["videos"]
    assert set(split["split_group"]) == {"train", "test"}
    assert set(split["split_time"]) <= {"train", "test", "excluded"}
    channels = videos["channel_id"].groupby(split["split_group"]).agg(set)
    assert not channels["train"] & channels["test"]
