"""
utils.py — Helper functions shared by the Lab 4 notebooks (IMDb review labeling).
Author: Kshitij Maurya
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

HTML_TAG = re.compile(r"<[^>]+>")
RATING_PATTERN = re.compile(r"\b(\d{1,2}(?:\.\d)?)\s*(?:/|out of)\s*10\b")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "this", "that", "was",
    "i", "for", "with", "as", "on", "but", "his", "her", "he", "she", "be", "are", "you",
    "they", "at", "by", "have", "has", "from", "who", "one", "all", "its", "so", "there",
    "about", "film", "movie", "just", "what", "their", "when", "which", "some", "if",
}


# ---------------------------------------------------------------- file helpers

def load_reviews(path: PathLike) -> Optional[pd.DataFrame]:
    """Read a reviews CSV. Returns None instead of raising if the file is missing."""
    path = Path(path)
    if not path.exists():
        logger.error("File not found: %s", path)
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    print(f"Loaded {path.name}: {len(df):,} rows, {df.shape[1]} columns")
    return df


def save_csv(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to CSV, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df):,} rows -> {path.name}")


def save_fig(fig, path: PathLike) -> None:
    """Save a matplotlib figure, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- text helpers

def clean_review(text) -> str:
    """Remove HTML tags such as <br />, lowercase, and squeeze whitespace."""
    if pd.isna(text):
        return ""
    text = HTML_TAG.sub(" ", str(text)).lower()
    return " ".join(text.split())


def word_count(text) -> int:
    """Number of whitespace-separated words (0 for missing text)."""
    if pd.isna(text):
        return 0
    return len(str(text).split())


def extract_rating(text) -> Optional[float]:
    """Return the last 'x/10' or 'x out of 10' score mentioned in a review, if any."""
    if pd.isna(text):
        return None
    scores = [float(s) for s in RATING_PATTERN.findall(str(text)) if float(s) <= 10]
    return scores[-1] if scores else None


def contains_any(text, words: List[str]) -> bool:
    """True if any of the words/phrases occurs in the text as a whole word."""
    if pd.isna(text) or not words:
        return False
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    return re.search(pattern, str(text).lower()) is not None


def top_words(texts: pd.Series, n: int = 15) -> pd.DataFrame:
    """Most frequent non-stopword words across a series of cleaned texts."""
    counts: Counter = Counter()
    for t in texts.fillna(""):
        counts.update(w for w in re.findall(r"[a-z']+", t) if w not in STOPWORDS and len(w) > 2)
    return pd.DataFrame(counts.most_common(n), columns=["word", "count"])
