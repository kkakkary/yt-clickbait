import pandas as pd

from ytclickbait.data.language import add_language_flags


def test_flags_english_and_french():
    df = pd.DataFrame(
        {
            "title": [
                "I tried the hardest workout in the world for thirty days",
                "Je suis allé au marché ce matin avec ma famille",
                "😱🔥 #shorts",
                # Name-heavy English titles get low English confidence but must be kept.
                "MrBeast | $1 vs $1,000,000 Hotel",
                "Minecraft Hardcore Ep 4",
                "Bad Bunny - Tití Me Preguntó",
                # Short non-Latin title must not slip through the short-title exemption.
                "今日の動画",
            ]
        }
    )
    out = add_language_flags(df)
    assert out["is_english"].tolist() == [True, False, True, True, True, False, False]
    assert out["english_confidence"].between(0, 1).all()
    assert out.loc[1, "detected_language"] == "french"
