"""Hand-crafted clickbait features. Each is easy to read off a title and easy to explain."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import emoji
import pandas as pd

WORDS_FILE = Path(__file__).resolve().parents[1] / "resources" / "clickbait_words.txt"

_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_NUMBER = re.compile(r"\d")


@lru_cache(maxsize=1)
def clickbait_pattern(path: Path = WORDS_FILE) -> re.Pattern[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    phrases = [ln.strip().lower() for ln in lines if ln.strip() and not ln.startswith("#")]
    # Longest first so "you won't believe" wins over "won't believe".
    phrases.sort(key=len, reverse=True)
    alts = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)


def title_features(title: str) -> dict[str, float]:
    letters = [c for c in title if c.isalpha()]
    words = _WORD.findall(title)
    caps_words = [w for w in words if len(w) >= 2 and w.isupper()]
    stripped = title.lstrip()
    return {
        "n_chars": len(title),
        "n_words": len(title.split()),
        "caps_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "n_caps_words": len(caps_words),
        "n_exclaim": title.count("!"),
        "n_question": title.count("?"),
        "has_number": bool(_NUMBER.search(title)),
        "starts_with_number": stripped[:1].isdigit(),
        "n_emoji": emoji.emoji_count(title),
        "has_brackets": any(c in title for c in "[]()"),
        "has_pipe_or_colon": any(c in title for c in "|:"),
        "has_hashtag": "#" in title,
        "has_second_person": bool(re.search(r"\byou(r|'re)?\b", title, re.IGNORECASE)),
        # Curly apostrophes are common in titles; the word list uses straight ones.
        "n_clickbait_words": len(clickbait_pattern().findall(title.replace("’", "'"))),
    }


def rule_features(titles: pd.Series) -> pd.DataFrame:
    feats = pd.DataFrame([title_features(t) for t in titles], index=titles.index)
    bool_cols = feats.select_dtypes(bool).columns
    feats[bool_cols] = feats[bool_cols].astype(int)
    return feats
