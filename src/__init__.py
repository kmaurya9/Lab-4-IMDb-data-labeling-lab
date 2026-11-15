"""Lab 4 — IMDb review data labeling: shared helpers."""

from .config import lab_path, load_config
from .data_validator import DataValidator
from .utils import (
    clean_review,
    contains_any,
    extract_rating,
    load_reviews,
    save_csv,
    save_fig,
    top_words,
    word_count,
)
