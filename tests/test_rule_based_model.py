import pandas as pd
import pytest

from ytclickbait.rule_based_model_inputs import merge_rule_inputs, score_rule_predictions


@pytest.fixture
def tables():
    """Create fresh sample data for videos A and B for each test."""
    videos = pd.DataFrame({
        "video_id": ["A", "B"],
        "title": ["hello", "WOW!!"],
    })

    # List the videos in reverse order to test whether the merge matches by ID.
    split = pd.DataFrame({
        "video_id": ["B", "A"],
        "split_group": ["test", "train"],
    })

    # The rule features also list B first. Only B has an all-caps title
    # with two exclamation marks.
    rules = pd.DataFrame({
        "video_id": ["B", "A"],
        "n_chars": [5, 5],
        "n_words": [1, 1],
        "caps_ratio": [1.0, 0.0],
        "n_caps_words": [1, 0],
        "n_exclaim": [2, 0],
        "n_question": [0, 0],
        "has_number": [0, 0],
        "starts_with_number": [0, 0],
        "n_emoji": [0, 0],
        "has_brackets": [0, 0],
        "has_pipe_or_colon": [0, 0],
        "has_hashtag": [0, 0],
        "has_second_person": [0, 0],
        "n_clickbait_words": [0, 0],
    })
    return videos, split, rules


def test_merge_matches_each_video_by_id(tables):
    # Each video should keep its own split and features regardless of row order.
    videos, split, rules = tables
    result = merge_rule_inputs(videos, split, rules).set_index("video_id")

    assert len(result) == 2
    assert result.index.is_unique
    assert result["split_group"].to_dict() == {"A": "train", "B": "test"}
    assert result["n_exclaim"].to_dict() == {"A": 0, "B": 2}
    assert result["caps_ratio"].to_dict() == {"A": 0.0, "B": 1.0}


@pytest.mark.parametrize("bad_id, expected_error", [
    ("B", "duplicates"),          # Both rows in rules now have ID B.
    (None, "missing values"),     # The second row has no video ID.
    ("C", "video_id mismatch"),   # The row count stays the same, but C replaces A.
])
def test_rejects_invalid_video_ids(tables, bad_id, expected_error):
    videos, split, rules = tables
    rules.loc[1, "video_id"] = bad_id

    # The function must raise the expected error for each invalid input.
    with pytest.raises(ValueError, match=expected_error):
        merge_rule_inputs(videos, split, rules)


@pytest.mark.parametrize("bad_values, expected_error", [
    ([1.2, 0.0], "between 0 and 1"),  # One ratio exceeds 1.
    ([1.0, None], "missing values"),  # One video has no feature value.
    (["HIGH", "LOW"], "numeric"),     # The numeric column contains strings.
])
def test_rejects_invalid_feature_values(tables, bad_values, expected_error):
    videos, split, rules = tables
    rules["caps_ratio"] = bad_values

    with pytest.raises(ValueError, match=expected_error):
        merge_rule_inputs(videos, split, rules)


def test_rejects_missing_rule_feature(tables):
    videos, split, rules = tables
    rules = rules.drop(columns="n_exclaim")  # Remove one of the 14 rule features.

    with pytest.raises(ValueError, match="Expected 14 rule features"):
        merge_rule_inputs(videos, split, rules)


def test_score_rule_predictions_known_values():
    """Check RMSE and R² against values calculated by hand."""
    scores = score_rule_predictions(y_true=[1, 5], y_pred=[1, 3])

    assert scores["RMSE"] == pytest.approx(2 ** 0.5)
    assert scores["R²"] == pytest.approx(0.5)