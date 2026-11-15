"""
data_validator.py — Simple data-quality checks run before each pipeline step.
Author: Kshitij Maurya
"""

from typing import List, Optional

import pandas as pd


class DataValidator:
    """
    Runs a fixed set of checks on a DataFrame and collects (name, passed, message) rows.

    Example
    -------
    >>> v = DataValidator(required_columns=["review", "sentiment"], label_col="sentiment")
    >>> results = v.run(df)
    >>> v.print_report()
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        text_col: Optional[str] = None,
        min_words: int = 3,
        label_col: Optional[str] = None,
        min_class_share: float = 0.10,
    ):
        self.required_columns = required_columns or []
        self.min_rows = min_rows
        self.text_col = text_col
        self.min_words = min_words
        self.label_col = label_col
        self.min_class_share = min_class_share
        self.results: List[dict] = []

    def _record(self, name: str, passed: bool, message: str) -> None:
        self.results.append({"check": name, "passed": bool(passed), "message": message})

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run every check and return the results as a DataFrame."""
        self.results = []

        self._record("enough rows", len(df) >= self.min_rows, f"{len(df):,} rows (need {self.min_rows:,})")

        missing = [c for c in self.required_columns if c not in df.columns]
        self._record("required columns", not missing, "all present" if not missing else f"missing {missing}")

        n_null = int(df.isna().sum().sum())
        self._record("no missing values", n_null == 0, f"{n_null} missing cells")

        n_dup = int(df.duplicated().sum())
        self._record("no duplicate rows", n_dup == 0, f"{n_dup} duplicates")

        if self.text_col in df.columns:
            words = df[self.text_col].fillna("").astype(str).str.split().str.len()
            n_short = int((words < self.min_words).sum())
            self._record("text long enough", n_short == 0, f"{n_short} texts under {self.min_words} words")

        if self.label_col in df.columns:
            shares = df[self.label_col].value_counts(normalize=True)
            ok = len(shares) >= 2 and shares.min() >= self.min_class_share
            self._record("class balance", ok, ", ".join(f"{k}: {v:.1%}" for k, v in shares.items()))

        return pd.DataFrame(self.results)

    @property
    def passed(self) -> bool:
        return all(r["passed"] for r in self.results)

    def print_report(self) -> None:
        status = "PASSED" if self.passed else "SOME CHECKS FAILED"
        print(f"Data validation: {status}")
        for r in self.results:
            print(f"  [{'ok' if r['passed'] else 'X '}] {r['check']:<20} {r['message']}")
