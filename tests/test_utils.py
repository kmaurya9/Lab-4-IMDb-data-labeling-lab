"""
test_utils.py — Tests for src/utils.py
Author: Kshitij Maurya
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    clean_review,
    contains_any,
    extract_rating,
    load_reviews,
    save_csv,
    top_words,
    word_count,
)


def test_clean_review_strips_html_and_lowercases():
    assert clean_review("Great <br /><br />MOVIE!") == "great movie!"


def test_clean_review_handles_missing():
    assert clean_review(np.nan) == ""


def test_word_count():
    assert word_count("one two  three") == 3
    assert word_count(None) == 0


def test_extract_rating_slash_and_out_of():
    assert extract_rating("i give it 8/10") == 8.0
    assert extract_rating("solid 7 out of 10") == 7.0


def test_extract_rating_uses_last_score():
    assert extract_rating("others said 9/10 but for me 3/10") == 3.0


def test_extract_rating_ignores_invalid_or_missing():
    assert extract_rating("no score here") is None
    assert extract_rating("15/10 would watch") is None
    assert extract_rating(None) is None


def test_contains_any_matches_whole_words_only():
    assert contains_any("what a waste of time", ["waste of"])
    assert not contains_any("the boringness", ["boring"])
    assert not contains_any(None, ["boring"])


def test_top_words_skips_stopwords():
    result = top_words(pd.Series(["the plot was great", "great acting and great plot"]), n=2)
    assert result.iloc[0]["word"] == "great"
    assert result.iloc[0]["count"] == 3
    assert "the" not in result["word"].tolist()


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "sub" / "reviews.csv"
    save_csv(pd.DataFrame({"review": ["a", "b"], "sentiment": ["positive", "negative"]}), path)
    df = load_reviews(path)
    assert df is not None and len(df) == 2


def test_load_missing_file_returns_none(tmp_path):
    assert load_reviews(tmp_path / "missing.csv") is None
