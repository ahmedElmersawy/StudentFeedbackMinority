"""
Unit tests for Feedback Atlas core pipeline functions.

Run with:
  cd /path/to/StudentFeedbackMinority
  pytest tests/test_pipeline.py -v
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df():
    """Tiny DataFrame with text + rating."""
    return pd.DataFrame({
        "feedback": [
            "This course was excellent, the professor explained everything clearly.",
            "Terrible class. Wasted my time completely. Very disappointing.",
            "It was okay, not great but not terrible either.",
            "I struggled with the workload but the content was good.",
            "Amazing lectures! Highly recommend this professor.",
        ],
        "rating": [5.0, 1.0, 3.0, 3.5, 5.0],
    })


@pytest.fixture()
def catme_csv_bytes():
    """Simulate a headerless CATME CSV (no column names)."""
    lines = [
        "Purva is awesome and always contributes to the team.",
        "Steven has a great attitude and helps with group tasks.",
        "I think my teammate could communicate better with the group.",
        "She was reliable and always showed up to meetings.",
        "He didn't respond to messages and missed three meetings.",
    ]
    return "\n".join(lines).encode("utf-8")


@pytest.fixture()
def labeled_csv_bytes():
    """CSV with explicit sentiment column."""
    df = pd.DataFrame({
        "comment": [
            "Great teaching style, very engaging.",
            "Horrible grading, completely unfair.",
            "Average experience overall.",
        ],
        "sentiment": ["Positive", "Negative", "Neutral"],
    })
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# backend.anonymizer
# ---------------------------------------------------------------------------

class TestAnonymizer:
    def test_replaces_person_name(self):
        from backend.anonymizer import anonymize_text
        # spaCy may not be installed in test env; regex fallback should still work
        result = anonymize_text("John Smith did a great job.")
        assert "John Smith" not in result

    def test_empty_string_passthrough(self):
        from backend.anonymizer import anonymize_text
        assert anonymize_text("") == ""
        assert anonymize_text("   ") == "   "

    def test_series_anonymization(self):
        from backend.anonymizer import anonymize_series
        s = pd.Series(["Alice Johnson helped the team.", "Good work overall."])
        result = anonymize_series(s)
        assert "Alice Johnson" not in result[0]
        assert result[1] == "Good work overall."  # No name to replace


# ---------------------------------------------------------------------------
# backend.mismatch_detector
# ---------------------------------------------------------------------------

class TestMismatchDetector:
    def test_high_mismatch_flagged(self):
        from backend.mismatch_detector import detect_mismatches
        df = pd.DataFrame({
            "text": ["Great professor but I still didn't like it much."],
            "prediction": ["Negative"],
            "rating": [5.0],
        })
        result = detect_mismatches(df, rating_col="rating")
        assert result.iloc[0]["mismatch_flag"] == True
        assert result.iloc[0]["mismatch_type"] == "HIGH_MISMATCH"

    def test_reverse_mismatch_flagged(self):
        from backend.mismatch_detector import detect_mismatches
        df = pd.DataFrame({
            "text": ["Wonderful experience, loved everything!"],
            "prediction": ["Positive"],
            "rating": [1.0],
        })
        result = detect_mismatches(df, rating_col="rating")
        assert result.iloc[0]["mismatch_flag"] == True
        assert result.iloc[0]["mismatch_type"] == "REVERSE_MISMATCH"

    def test_no_mismatch(self):
        from backend.mismatch_detector import detect_mismatches
        df = pd.DataFrame({
            "text": ["Great class!"],
            "prediction": ["Positive"],
            "rating": [5.0],
        })
        result = detect_mismatches(df, rating_col="rating")
        assert result.iloc[0]["mismatch_flag"] == False

    def test_missing_rating_col_graceful(self):
        from backend.mismatch_detector import detect_mismatches
        df = pd.DataFrame({"text": ["Good."], "prediction": ["Positive"]})
        result = detect_mismatches(df, rating_col=None)
        assert "mismatch_flag" in result.columns
        assert result["mismatch_flag"].sum() == 0

    def test_nan_rating_skipped(self):
        from backend.mismatch_detector import detect_mismatches
        df = pd.DataFrame({
            "text": ["Some feedback."],
            "prediction": ["Positive"],
            "rating": [float("nan")],
        })
        result = detect_mismatches(df, rating_col="rating")
        assert result.iloc[0]["mismatch_flag"] == False


# ---------------------------------------------------------------------------
# backend.pipeline — column detection
# ---------------------------------------------------------------------------

class TestColumnDetection:
    def test_detects_feedback_col(self):
        from backend.pipeline import auto_detect_text_columns
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "feedback": ["Great class with excellent professor.", "Boring lectures.", "Okay."],
            "rating": [5, 1, 3],
        })
        cols = auto_detect_text_columns(df)
        assert "feedback" in cols
        assert "rating" not in cols

    def test_detects_rating_col(self):
        from backend.pipeline import _detect_rating_column
        df = pd.DataFrame({
            "feedback": ["Some text here that is long enough."],
            "score": [4.5],
        })
        col = _detect_rating_column(df, text_cols=["feedback"])
        assert col == "score"

    def test_headerless_detection(self):
        from backend.pipeline import _looks_like_headerless
        assert _looks_like_headerless("Purva is awesome and always contributes to the team")
        assert not _looks_like_headerless("feedback_text")
        assert not _looks_like_headerless("Q1_Comments")


# ---------------------------------------------------------------------------
# backend.pipeline — ingest_csv
# ---------------------------------------------------------------------------

class TestIngestCsv:
    def test_headerless_catme(self, catme_csv_bytes):
        from backend.pipeline import ingest_csv
        df, text_cols, rating_col, label_col = ingest_csv(catme_csv_bytes, anonymize=False)
        assert "text" in df.columns
        assert len(df) == 5
        assert text_cols == ["text"]
        assert rating_col is None

    def test_labeled_csv(self, labeled_csv_bytes):
        from backend.pipeline import ingest_csv
        df, text_cols, rating_col, label_col = ingest_csv(labeled_csv_bytes, anonymize=False)
        assert "text" in df.columns
        assert label_col == "sentiment"
        assert len(df) == 3

    def test_rated_csv(self, simple_df):
        from backend.pipeline import ingest_csv
        buf = io.StringIO()
        simple_df.to_csv(buf, index=False)
        df, text_cols, rating_col, label_col = ingest_csv(buf.getvalue().encode(), anonymize=False)
        assert "feedback" in text_cols
        assert rating_col == "rating"


# ---------------------------------------------------------------------------
# training.train — column detection
# ---------------------------------------------------------------------------

class TestTrainColumnDetection:
    def test_auto_detect_text(self, simple_df):
        from training.train import auto_detect_text_columns
        cols = auto_detect_text_columns(simple_df)
        assert "feedback" in cols

    def test_auto_detect_rating(self, simple_df):
        from training.train import auto_detect_rating_column
        col = auto_detect_rating_column(simple_df, text_cols=["feedback"])
        assert col == "rating"

    def test_derive_labels_basic(self, simple_df):
        from training.train import derive_labels_from_ratings
        labels = derive_labels_from_ratings(simple_df, "rating")
        assert set(labels.dropna().unique()).issubset({"Positive", "Neutral", "Negative"})
        assert labels.iloc[0] == "Positive"  # rating 5.0
        assert labels.iloc[1] == "Negative"  # rating 1.0

    def test_headerless_csv_loading(self, catme_csv_bytes, tmp_path):
        from training.train import load_csv
        f = tmp_path / "catme.csv"
        f.write_bytes(catme_csv_bytes)
        df = load_csv(f)
        assert "text" in df.columns
        assert len(df) == 5


# ---------------------------------------------------------------------------
# backend.minority_detector — keyword matching
# ---------------------------------------------------------------------------

class TestMinorityDetector:
    def test_keyword_match(self):
        from backend.minority_detector import _keyword_match
        kw_map = {
            "Mental_Health": ["anxiety", "burnout", "overwhelmed"],
            "Financial_Hardship": ["financial aid", "can't afford"],
        }
        assert "Mental_Health" in _keyword_match("I feel overwhelmed by the workload", kw_map)
        assert "Financial_Hardship" in _keyword_match("I applied for financial aid this semester", kw_map)
        assert _keyword_match("Great course, enjoyed it.", kw_map) == []

    def test_category_breakdown(self):
        from backend.minority_detector import category_breakdown
        df = pd.DataFrame({
            "minority_category": [
                "Mental_Health|Financial_Hardship",
                "Mental_Health",
                "",
                "Caregiver",
                "Statistical_Outlier_Only",
            ]
        })
        breakdown = category_breakdown(df)
        assert breakdown["Mental_Health"] == 2
        assert breakdown["Financial_Hardship"] == 1
        assert "Caregiver" in breakdown
