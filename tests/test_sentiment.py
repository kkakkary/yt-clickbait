import pandas as pd
import pytest

from ytclickbait.features.sentiment import roberta_scores, vader_scores

TITLES = pd.Series(
    [
        "This is the best day of my life, I love it!",
        "Worst experience ever, I hate this so much",
        "Episode 4 trailer",
    ]
)


def test_vader_signs():
    s = vader_scores(TITLES)
    assert s.loc[0, "vader_compound"] > 0.5
    assert s.loc[1, "vader_compound"] < -0.5
    assert abs(s.loc[2, "vader_compound"]) < 0.2


def test_vader_reads_caps_and_exclamation():
    s = vader_scores(pd.Series(["this is good", "this is GOOD!!!"]))
    assert s.loc[1, "vader_compound"] > s.loc[0, "vader_compound"]


@pytest.mark.slow
def test_roberta_signs():
    s = roberta_scores(TITLES, batch_size=2)
    assert {"roberta_negative", "roberta_neutral", "roberta_positive"} <= set(s.columns)
    assert s.loc[0, "roberta_score"] > 0.5
    assert s.loc[1, "roberta_score"] < -0.5
    assert (
        s[["roberta_negative", "roberta_neutral", "roberta_positive"]]
        .sum(axis=1)
        .round(4)
        .eq(1)
        .all()
    )
