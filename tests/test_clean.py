import numpy as np
import pandas as pd
import pytest

from ytclickbait.data.clean import clean
from ytclickbait.data.load import load_all


@pytest.fixture
def videos(raw_dir):
    raw, cats = load_all(raw_dir, ["US", "GB", "CA"])
    return clean(raw, cats).set_index("video_id")


def test_one_row_per_video_and_zero_views_dropped(videos):
    assert videos.index.is_unique
    assert set(videos.index) == {"v1", "v2", "v3"}


def test_keeps_latest_snapshot_across_countries(videos):
    v1 = videos.loc["v1"]
    assert v1["view_count"] == 9000
    assert v1["snapshot_country"] == "GB"
    assert v1["title"] == "You Won't Believe This"
    assert v1["countries"] == ["GB", "US"]
    assert v1["n_countries"] == 2
    assert v1["days_trending"] == 3


def test_title_history(videos):
    v1 = videos.loc["v1"]
    assert v1["n_distinct_titles"] == 2
    assert v1["title_changed"]
    assert v1["first_title"] == "YOU WON'T BELIEVE THIS!!!"
    assert v1["last_title"] == "You Won't Believe This"
    assert not videos.loc["v2", "title_changed"]


def test_categories_mapped(videos):
    assert videos.loc["v1", "category_name"] == "Entertainment"
    assert videos.loc["v2", "category_name"] == "Music"


def test_age_and_targets(videos):
    v1 = videos.loc["v1"]
    assert v1["age_days"] == pytest.approx(2 + 14 / 24)
    assert v1["log_views"] == pytest.approx(np.log1p(9000))
    assert v1["likes_per_view"] == pytest.approx(700 / 9000)
    assert v1["n_tags"] == 3
    assert videos.loc["v2", "n_tags"] == 0


def test_age_floor_is_one_day(videos):
    assert (videos["age_days"] >= 1).all()


def test_unparseable_dates_raise(raw_dir):
    path = raw_dir / "US_youtube_trending_data.csv"
    df = pd.read_csv(path)
    df["trending_date"] = "17.14.11"  # old yy.dd.mm format
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="trending_date"):
        load_all(raw_dir, ["US", "GB", "CA"])


def test_disabled_engagement_is_nan_not_zero(videos):
    assert np.isnan(videos.loc["v2", "comments_per_view"])
    assert not np.isnan(videos.loc["v2", "likes_per_view"])
    assert np.isnan(videos.loc["v3", "likes_per_view"])
