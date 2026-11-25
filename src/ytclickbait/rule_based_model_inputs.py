import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import mean_squared_error, r2_score


def merge_rule_inputs(videos, split, rules):
    """Validate the model inputs for each video and merge them by video_id."""

    # 1. Each table must have one unique, non-missing video_id per video.
    for name, table in [("videos", videos), ("split", split), ("rules", rules)]:
        if "video_id" not in table.columns:
            raise ValueError(f"{name}: missing video_id column.")
        if table["video_id"].isna().any():
            raise ValueError(f"{name}: video_id contains missing values.")
        if not table["video_id"].is_unique:
            raise ValueError(f"{name}: video_id contains duplicates.")

    # 2. Compare ID sets because tables with the same row count can contain different videos.
    video_ids = set(videos["video_id"])
    for name, table in [("split", split), ("rules", rules)]:
        other_ids = set(table["video_id"])
        if other_ids != video_ids:
            raise ValueError(
                f"{name}: video_id mismatch "
                f"({len(video_ids - other_ids)} missing, "
                f"{len(other_ids - video_ids)} extra)."
            )

    # 3. Check that there are 14 rule features and each is numeric with no missing values.
    rule_cols = rules.columns.drop("video_id")
    if len(rule_cols) != 14:
        raise ValueError(f"Expected 14 rule features, found {len(rule_cols)}.")

    for col in rule_cols:
        if not is_numeric_dtype(rules[col]):
            raise ValueError(f"{col}: expected a numeric column.")
        if rules[col].isna().any():
            raise ValueError(f"{col}: contains missing values.")

    # 4. The uppercase ratio must be between 0 and 1, inclusive.
    if "caps_ratio" not in rules or not rules["caps_ratio"].between(0, 1).all():
        raise ValueError("caps_ratio: expected a ratio column with values between 0 and 1.")

    # 5. Use videos as the base table and match rows by video_id regardless of row order.
    return (
        videos
        .merge(split, on="video_id", how="left", validate="one_to_one")
        .merge(rules, on="video_id", how="left", validate="one_to_one")
    )

def score_rule_predictions(y_true, y_pred):
    """Calculate RMSE and R² for predicted log views."""
    return {
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R²": r2_score(y_true, y_pred),
    }