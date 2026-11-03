"""Flag English titles so the word list, VADER, and TF-IDF stay valid.

The statistical detector (lingua) alone is unreliable on short titles full of names and
caps: on real data it called "KSI TALENT SHOW" Tagalog and "SIDEMEN RAGE AT IMPOSSIBLE
FALL GUYS MAP" French. So a title is dropped only with real evidence it isn't English:

- mostly non-Latin script (Korean, Hindi, Arabic, ...), or
- lingua says non-English AND the title has common French/Spanish function words ("le",
  "de", "j'ai", "oficial") or at least 20% of its words carry accents.

Everything else is kept. A fixed English-confidence cutoff was tried first and rejected:
it dropped most name-heavy English titles.
"""

from __future__ import annotations

import re
import unicodedata

import emoji
import pandas as pd
from lingua import Language, LanguageDetectorBuilder

_NOISE = re.compile(r"https?://\S+|[#@]\w+|\d+")

CANDIDATE_LANGUAGES = [
    "ENGLISH", "FRENCH", "SPANISH", "PORTUGUESE", "GERMAN", "ITALIAN", "DUTCH", "POLISH",
    "TURKISH", "RUSSIAN", "ARABIC", "HINDI", "PUNJABI", "URDU", "TAGALOG", "INDONESIAN",
    "VIETNAMESE", "KOREAN", "JAPANESE", "CHINESE",
]  # fmt: skip

# Common French/Spanish/Portuguese words that rarely appear in English titles.
FOREIGN_WORDS = {
    "le", "la", "les", "des", "du", "une", "et", "est", "avec", "dans", "pour", "qui", "sur",
    "pas", "mon", "mes", "je", "j'ai", "c'est", "nous", "vous", "au", "aux", "officiel",
    "officielle", "el", "los", "las", "del", "con", "por", "para", "que", "y", "mi", "tu",
    "su", "es", "oficial", "resumen", "de", "um", "uma", "com", "não",
}  # fmt: skip
ACCENTED_WORD_SHARE = 0.2
_WORDS = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")


def _letters_only(title: str) -> str:
    """Drop emojis, URLs, hashtags, and digits, which carry no language signal."""
    return _NOISE.sub(" ", emoji.replace_emoji(title, replace=" ")).strip()


def _is_latin(c: str) -> bool:
    """Latin letters, including accented ones like é or ñ."""
    return unicodedata.name(c, "").startswith("LATIN")


def detect_languages(
    titles: pd.Series, candidates: list[str] = CANDIDATE_LANGUAGES
) -> pd.DataFrame:
    langs = [getattr(Language, name) for name in candidates]
    detector = (
        LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()
    )
    texts = [_letters_only(t) for t in titles]
    detected = detector.detect_languages_in_parallel_of(texts)
    confidence = detector.compute_language_confidence_in_parallel(texts, Language.ENGLISH)
    return pd.DataFrame(
        {
            "detected_language": [d.name.lower() if d else "unknown" for d in detected],
            "english_confidence": confidence,
        },
        index=titles.index,
    )


def _script_and_word_evidence(title: str) -> dict[str, float | bool]:
    text = _letters_only(title).replace("’", "'")
    letters = [c for c in text if c.isalpha()]
    latin = [c for c in letters if _is_latin(c)]
    words = _WORDS.findall(text)
    accented = [w for w in words if any(not c.isascii() and _is_latin(c) for c in w)]
    return {
        "n_latin": len(latin),
        "non_latin_share": 1 - len(latin) / len(letters) if letters else 0.0,
        # Share of words, not any accent: one name like "Pokémon" shouldn't count.
        "has_accent": len(accented) / len(words) >= ACCENTED_WORD_SHARE if words else False,
        "has_foreign_word": bool({w.lower() for w in words} & FOREIGN_WORDS),
    }


def add_language_flags(
    df: pd.DataFrame, min_letters: int = 10, candidates: list[str] = CANDIDATE_LANGUAGES
) -> pd.DataFrame:
    """Add detected_language, english_confidence, non_english_reason, is_english."""
    out = df.copy()
    ev = pd.DataFrame([_script_and_word_evidence(t) for t in out["title"]], index=out.index)
    out = out.join(detect_languages(out["title"], candidates))

    few_latin = ev["n_latin"] < min_letters
    some_non_latin = ev["non_latin_share"] > 0
    lingua_foreign = out["detected_language"] != "english"

    # Mostly non-Latin, or a short title with any non-Latin script (e.g. "今日の動画").
    non_latin = (ev["non_latin_share"] > 0.5) | (some_non_latin & few_latin)
    # Short Latin-only titles are too short to judge, so they are kept.
    foreign = lingua_foreign & (ev["has_accent"] | ev["has_foreign_word"]) & ~few_latin

    reason = pd.Series("", index=out.index)
    reason[foreign] = "foreign words or accents"
    reason[non_latin] = "non-Latin script"
    out["non_english_reason"] = reason
    out["is_english"] = reason == ""
    return out
