"""Flag English titles so the word list, VADER, and TF-IDF stay valid.

A title is kept as English when English is the most likely language among a short list
of languages common on US/GB/CA trending. A fixed confidence cutoff doesn't work: short
titles full of names ("MrBeast | $1 vs $1,000,000 Hotel") get low English confidence
even though they are English, because the probability is spread over many languages.
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


def _letters_only(title: str) -> str:
    """Drop emojis, URLs, hashtags, and digits, which carry no language signal."""
    return _NOISE.sub(" ", emoji.replace_emoji(title, replace=" ")).strip()


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


def add_language_flags(
    df: pd.DataFrame, min_letters: int = 10, candidates: list[str] = CANDIDATE_LANGUAGES
) -> pd.DataFrame:
    """Add detected_language, english_confidence, is_english. Latin-script titles with too
    few letters can't be classified, so they count as English. Any non-Latin letter
    (e.g. a short Japanese title) rules that exemption out."""
    out = df.copy()
    letters = out["title"].map(lambda t: [c for c in _letters_only(t) if c.isalpha()])
    n_latin = letters.map(lambda cs: sum(_is_latin(c) for c in cs))
    n_other = letters.str.len() - n_latin
    out = out.join(detect_languages(out["title"], candidates))
    too_short = (n_latin < min_letters) & (n_other == 0)
    out["is_english"] = too_short | (out["detected_language"] == "english")
    return out


def _is_latin(c: str) -> bool:
    """Latin letters, including accented ones like é or ñ."""
    return unicodedata.name(c, "").startswith("LATIN")
