"""Mismatch detection: flag rows where numeric rating contradicts text sentiment."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _cfg() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _broad(label: str) -> str:
    """Collapse any fine-grained label to Positive / Neutral / Negative."""
    l = label.lower()
    if l.startswith("positive") or l == "positive":
        return "Positive"
    if l.startswith("negative") or l == "negative":
        return "Negative"
    return "Neutral"


def detect_mismatches(
    df: pd.DataFrame,
    prediction_col: str = "prediction",
    rating_col: str | None = None,
    high_rating_threshold: float | None = None,
    low_rating_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Add ``mismatch_flag`` (bool) and ``mismatch_type`` (str) columns.

    HIGH_MISMATCH    — rating ≥ high_threshold but sentiment is Negative/Neutral.
    REVERSE_MISMATCH — rating ≤ low_threshold  but sentiment is Positive.
    """
    mm_cfg = _cfg().get("mismatch", {})
    high_thresh = high_rating_threshold if high_rating_threshold is not None else mm_cfg.get("high_rating_threshold", 3.8)
    low_thresh = low_rating_threshold if low_rating_threshold is not None else mm_cfg.get("low_rating_threshold", 2.5)

    out = df.copy()
    out["mismatch_flag"] = False
    out["mismatch_type"] = ""

    if prediction_col not in out.columns:
        logger.warning("prediction column '%s' not found; skipping mismatch detection.", prediction_col)
        return out

    if rating_col is None or rating_col not in out.columns:
        logger.info("No rating column available; mismatch detection skipped.")
        return out

    ratings = pd.to_numeric(out[rating_col], errors="coerce")
    flag_col = out.columns.get_loc("mismatch_flag")
    type_col = out.columns.get_loc("mismatch_type")

    for i, (rating, pred_raw) in enumerate(zip(ratings, out[prediction_col].astype(str))):
        if pd.isna(rating):
            continue
        pred = _broad(pred_raw)
        if rating >= high_thresh and pred in ("Negative", "Neutral"):
            out.iloc[i, flag_col] = True
            out.iloc[i, type_col] = "HIGH_MISMATCH"
        elif rating <= low_thresh and pred == "Positive":
            out.iloc[i, flag_col] = True
            out.iloc[i, type_col] = "REVERSE_MISMATCH"

    n = int(out["mismatch_flag"].sum())
    logger.info("Mismatch detection: %d mismatches (%.1f%%)", n, 100.0 * n / max(1, len(out)))
    return out
