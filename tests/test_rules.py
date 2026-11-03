import pandas as pd
import pytest

from ytclickbait.features.rules import rule_features, title_features


def test_clickbaity_title():
    f = title_features("YOU WON'T BELIEVE THIS!!! 😱 10 tips?")
    assert f["n_exclaim"] == 3
    assert f["n_question"] == 1
    assert f["n_emoji"] == 1
    assert f["has_number"] is True
    assert f["starts_with_number"] is False
    assert f["n_caps_words"] == 4  # YOU, WON'T, BELIEVE, THIS
    assert f["caps_ratio"] == pytest.approx(18 / 22)
    assert f["n_clickbait_words"] == 1  # "you won't believe", counted once
    assert f["has_second_person"] is True


def test_plain_title():
    f = title_features("Calm piano music for studying")
    assert f["n_clickbait_words"] == 0
    assert f["n_caps_words"] == 0
    assert f["n_exclaim"] == 0
    assert not f["has_number"]


def test_curly_apostrophe_and_word_boundaries():
    assert title_features("You won’t believe it")["n_clickbait_words"] == 1
    # "best" inside "bestseller" is not a match
    assert title_features("A bestseller review")["n_clickbait_words"] == 0


def test_misc_flags():
    f = title_features("10 Hacks | Official (Lyrics) #shorts")
    assert (
        f["starts_with_number"]
        and f["has_brackets"]
        and f["has_pipe_or_colon"]
        and f["has_hashtag"]
    )


def test_empty_title():
    f = title_features("")
    assert f["n_chars"] == 0 and f["caps_ratio"] == 0.0


def test_dataframe_is_numeric_and_aligned():
    titles = pd.Series(["A!", "b?"], index=[5, 9])
    df = rule_features(titles)
    assert list(df.index) == [5, 9]
    assert all(pd.api.types.is_numeric_dtype(t) for t in df.dtypes)
