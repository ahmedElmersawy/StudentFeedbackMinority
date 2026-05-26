"""
Dataset loaders for Feedback Atlas dual-mode training.

Three sources:
  load_catme()      → CATMEcomments_Training.csv  (Student→Student, headerless)
  load_courseeval() → studentdataset.csv           (Student→Professor, Purdue)
  load_rmp()        → courseEval.csv               (Student→Professor, RMP ~8M)
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CATME — Student→Student peer feedback
# ---------------------------------------------------------------------------

def load_catme(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load CATME headerless CSV.
    File has no column header — first row is actual feedback text.
    Returns DataFrame with 'text', 'mode', 'source' columns.
    """
    path = Path(filepath or (_PROJECT_ROOT / "CATMEcomments_Training.csv"))
    df = pd.read_csv(path, header=None, names=["text"])
    df["mode"] = "student_to_student"
    df["source"] = "CATME"
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip().str.len() > 5]
    logger.info("[load_catme] Loaded %d rows from %s", len(df), path)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Course Eval — Student→Professor (Purdue studentdataset.csv)
# ---------------------------------------------------------------------------

# Mapping: (numeric_rating_col, text_col, dimension_name)
_COURSEEVAL_COLUMN_PAIRS = [
    ("teaching", "teaching.1", "teaching"),
    ("coursecontent", "coursecontent.1", "course_content"),
    ("examination", "Examination", "examination"),
    ("labwork", "labwork.1", "lab_work"),
    ("library_facilities", " library_facilities", "library"),
    ("extracurricular", "extracurricular.1", "extracurricular"),
]


def load_courseeval(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load Purdue course-eval dataset (paired rating + text columns).
    Melts all dimensions into one long DataFrame.
    Returns DataFrame with 'rating', 'text', 'dimension', 'mode', 'source'.
    """
    path = Path(filepath or (_PROJECT_ROOT / "studentdataset.csv"))
    df_raw = pd.read_csv(path)

    rows: list[pd.DataFrame] = []
    for rating_col, text_col, dimension in _COURSEEVAL_COLUMN_PAIRS:
        if rating_col not in df_raw.columns or text_col not in df_raw.columns:
            logger.warning("[load_courseeval] Missing columns: %s / %s — skipping.", rating_col, text_col)
            continue
        subset = df_raw[[rating_col, text_col]].copy()
        subset.columns = ["rating", "text"]
        subset["dimension"] = dimension
        subset = subset[subset["text"].notna()]
        subset = subset[subset["text"].str.strip().str.len() > 3]
        rows.append(subset)

    if not rows:
        raise ValueError(f"No valid column pairs found in {path}. Check column names.")

    result = pd.concat(rows, ignore_index=True)
    result["mode"] = "student_to_professor"
    result["source"] = "courseeval"
    logger.info("[load_courseeval] Loaded %d rows across %d dimensions from %s", len(result), len(rows), path)
    return result


# ---------------------------------------------------------------------------
# RMP — Rate My Professor (courseEval.csv / GitHub RMP dataset)
# ---------------------------------------------------------------------------

_RMP_LABEL_MAP = {
    "awesome": "Teaching_Positive_Engagement",
    "good": "Teaching_Positive_Clarity",
    "average": "Majority_Positive",
    "poor": "Teaching_Negative_Clarity",
    "awful": "Teaching_Negative_Clarity",
}


def _map_quality(q) -> int | None:
    """Map 1–5 quality rating to -1 / 0 / 1."""
    try:
        q = float(q)
        if q >= 4.0:
            return 1
        if q >= 3.0:
            return 0
        return -1
    except (ValueError, TypeError):
        return None


def load_rmp(
    filepath: str | Path | None = None,
    n_per_class: int = 50_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load Rate My Professor dataset.

    Keeps: comment (→ text), quality (→ rating -1/0/1), emotional_label, tags.
    Stratified-samples *n_per_class* rows per rating class (default 50k → 150k total).
    Returns DataFrame with 'text', 'rating', 'emotional_label', 'tags',
    'pseudo_label', 'dimension', 'mode', 'source'.
    """
    path = Path(filepath or (_PROJECT_ROOT / "courseEval.csv"))
    df = pd.read_csv(path, low_memory=False)

    needed = [c for c in ("quality", "difficulty", "comment", "emotional_label", "tags") if c in df.columns]
    df = df[needed].copy()
    if "comment" in df.columns:
        df = df.rename(columns={"comment": "text"})

    df["rating"] = df["quality"].apply(_map_quality)
    df = df[df["rating"].notna()].copy()
    df["rating"] = df["rating"].astype(int)
    df = df[df["text"].notna()]
    df = df[df["text"].str.strip().str.len() > 10]

    # Map emotional_label → pseudo label for zero-shot seed
    if "emotional_label" in df.columns:
        df["pseudo_label"] = df["emotional_label"].str.lower().str.strip().map(_RMP_LABEL_MAP)
    else:
        df["pseudo_label"] = None

    df["mode"] = "student_to_professor"
    df["source"] = "RMP"
    df["dimension"] = "overall"

    # Stratified sample
    before = len(df)
    df = (
        df.groupby("rating", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), n_per_class), random_state=random_state))
        .reset_index(drop=True)
    )
    logger.info(
        "[load_rmp] Loaded %d rows from %s → sampled %d (≤%d per class)",
        before, path, len(df), n_per_class,
    )
    return df


# ---------------------------------------------------------------------------
# Combined professor training set
# ---------------------------------------------------------------------------

def load_professor_combined(
    rmp_path: str | Path | None = None,
    courseeval_path: str | Path | None = None,
    rmp_n_per_class: int = 50_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df) where:
      train_df = RMP sample (for fine-tuning)
      val_df   = courseeval (held-out Purdue validation — do NOT train on)
    """
    train_df = load_rmp(rmp_path, n_per_class=rmp_n_per_class)
    val_df = load_courseeval(courseeval_path)
    return train_df, val_df
