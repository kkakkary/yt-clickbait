import numpy as np
import pandas as pd
import pytest

from ytclickbait.data.split import channel_baseline, group_split, within_channel_time_split


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    channels = np.repeat([f"c{i}" for i in range(20)], [15] * 5 + [3] * 15)
    return pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(len(channels))],
            "channel_id": channels,
            "published_at": pd.date_range("2021-01-01", periods=len(channels), freq="D"),
            "log_views": rng.normal(10, 1, len(channels)),
        }
    )


def test_group_split_has_no_shared_channels(df):
    split = group_split(df, test_size=0.25, seed=1)
    train = set(df.loc[split == "train", "channel_id"])
    test = set(df.loc[split == "test", "channel_id"])
    assert train and test and not train & test


def test_within_channel_split_tests_on_latest_videos(df):
    split = within_channel_time_split(df, min_videos=10, test_size=0.2)
    assert (split[df["channel_id"].isin([f"c{i}" for i in range(5, 20)])] == "excluded").all()
    for ch in [f"c{i}" for i in range(5)]:
        rows = df[df["channel_id"] == ch]
        s = split[rows.index]
        assert (s == "test").sum() == 3
        assert (
            rows.loc[s == "test", "published_at"].min()
            > rows.loc[s == "train", "published_at"].max()
        )


def test_channel_baseline_test_rows_and_fallback():
    df = pd.DataFrame(
        {
            "channel_id": ["a", "a", "a", "b", "c"],
            "log_views": [1.0, 2.0, 6.0, 10.0, 100.0],
        }
    )
    is_train = pd.Series([True, True, False, True, False])
    # Train rows in "a" depend on random fold assignment, so check only the rest.
    base = channel_baseline(df, is_train)
    assert base[2] == pytest.approx(1.5)  # test row: mean of train videos in "a"
    train_mean = np.mean([1.0, 2.0, 10.0])
    assert base[3] == pytest.approx(train_mean)  # only train video in "b": nothing out of fold
    assert base[4] == pytest.approx(train_mean)  # unseen channel


def test_channel_baseline_does_not_leak_own_target():
    """A train row's baseline must not depend on its own target value."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"channel_id": np.repeat(["a", "b"], 50), "log_views": rng.normal(10, 1, 100)}
    )
    is_train = pd.Series(True, index=df.index)
    before = channel_baseline(df, is_train)
    changed = df.copy()
    changed.loc[7, "log_views"] += 100
    after = channel_baseline(changed, is_train)
    assert after[7] == pytest.approx(before[7])
