"""Train/test splits and the channel baseline.

Two splits answer two different questions:
- split_group: no channel in both train and test. Does the model work for new creators?
- split_time: within each big channel, earlier videos train and later ones test. Holds the
  creator fixed so title effects are less mixed up with channel size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> pd.Series:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    _, test_idx = next(splitter.split(df, groups=df["channel_id"]))
    split = pd.Series("train", index=df.index, name="split_group")
    split.iloc[test_idx] = "test"
    return split


def within_channel_time_split(
    df: pd.DataFrame, min_videos: int = 10, test_size: float = 0.2
) -> pd.Series:
    """Latest test_size share of each channel's videos is test. Small channels are excluded."""
    split = pd.Series("excluded", index=df.index, name="split_time")
    sizes = df.groupby("channel_id")["video_id"].transform("size")
    big = df[sizes >= min_videos]
    rank = big.groupby("channel_id")["published_at"].rank(method="first", pct=True)
    split.loc[big.index] = np.where(rank > 1 - test_size, "test", "train")
    return split


def channel_baseline(
    df: pd.DataFrame,
    is_train: pd.Series,
    target: str = "log_views",
    n_folds: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Channel's typical target value, using only train rows.

    Train rows get an out-of-fold mean: the mean of their channel's train videos in the
    other folds. Leave-one-out is not used because (sum - y_i) / (n - 1) falls as y_i
    rises, so a model that can tell channels apart could read the target back out of it.
    Test rows get the mean of all their channel's train videos. With no usable channel
    videos, fall back to the overall train mean.

    Under split_group no test channel appears in train, so test rows always get the
    fallback; the baseline is only informative for split_time.
    """
    y = df[target].where(is_train)
    grp = df["channel_id"]
    fallback = y.mean()

    baseline = y.groupby(grp).transform("mean")  # test rows: full train mean of channel

    fold = pd.Series(np.random.default_rng(seed).integers(0, n_folds, len(df)), index=df.index)
    for k in range(n_folds):
        in_fold = is_train & (fold == k)
        others = y.where(fold != k)
        oof_mean = others.groupby(grp).transform("mean")
        baseline[in_fold] = oof_mean[in_fold]
    return baseline.fillna(fallback).rename(f"channel_baseline_{target}")
