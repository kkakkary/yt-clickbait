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
                # Real titles lingua alone misclassified as Tagalog / French / Polish.
                "KSI TALENT SHOW",
                "SIDEMEN RAGE AT IMPOSSIBLE FALL GUYS MAP",
                "CRAZIEST NO KILL WIN in WARZONE 2!",
                "Trailer 2 | Pinocchio | Disney+",
                # One accented name must not make an English title foreign.
                "100 YouTubers sent me a Pokémon for a Nuzlocke",
                # Real non-English titles with no accents.
                "Grupo Firme - La Cajetosa - (Official Video)",
                "Resumen de FC Barcelona vs Sevilla FC (1-0)",
                "[MV] IU(아이유)_LILAC(라일락)",
            ]
        }
    )
    out = add_language_flags(df)
    expected = [True, False, True, True, True, False, False]
    expected += [True, True, True, True, True, False, False, False]
    assert out["is_english"].tolist() == expected
    assert out["english_confidence"].between(0, 1).all()
    assert out.loc[1, "detected_language"] == "french"
