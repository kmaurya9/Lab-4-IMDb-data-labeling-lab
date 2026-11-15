"""
test_data_validator.py — Tests for src/data_validator.py
Author: Kshitij Maurya
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_validator import DataValidator  # noqa: E402


@pytest.fixture
def reviews():
    return pd.DataFrame({
        "review": [f"review number {i} has words" for i in range(20)],
        "sentiment": ["positive", "negative"] * 10,
    })


def check(results, name):
    return results.set_index("check").loc[name, "passed"]


def test_clean_data_passes(reviews):
    v = DataValidator(required_columns=["review", "sentiment"], min_rows=10,
                      text_col="review", label_col="sentiment")
    v.run(reviews)
    assert v.passed


def test_missing_column_fails(reviews):
    v = DataValidator(required_columns=["review", "rating"])
    assert not check(v.run(reviews), "required columns")


def test_too_few_rows_fails(reviews):
    v = DataValidator(min_rows=100)
    assert not check(v.run(reviews), "enough rows")


def test_duplicates_detected(reviews):
    v = DataValidator()
    assert not check(v.run(pd.concat([reviews, reviews.head(2)])), "no duplicate rows")


def test_missing_values_detected(reviews):
    reviews.loc[0, "review"] = None
    v = DataValidator()
    assert not check(v.run(reviews), "no missing values")


def test_short_text_detected(reviews):
    reviews.loc[0, "review"] = "bad"
    v = DataValidator(text_col="review", min_words=3)
    assert not check(v.run(reviews), "text long enough")


def test_imbalanced_labels_detected(reviews):
    reviews["sentiment"] = ["positive"] * 19 + ["negative"]
    v = DataValidator(label_col="sentiment", min_class_share=0.10)
    assert not check(v.run(reviews), "class balance")
