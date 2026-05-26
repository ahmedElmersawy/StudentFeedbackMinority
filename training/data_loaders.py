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
# Dimension-constrained zero-shot labels for courseeval
# ---------------------------------------------------------------------------

# Maps (dimension, rating_bucket) → subset of professor zero-shot candidate strings.
# Using a narrow candidate set per dimension makes BART-MNLI far more accurate
# than asking it to pick from all 24 labels at once.
_DIMENSION_CANDIDATES: dict[tuple[str, str], list[str]] = {
    ("teaching", "positive"):     ["professor explains clearly and teaches well",
                                   "professor is engaging and passionate"],
    ("teaching", "neutral"):      ["professor explains clearly and teaches well",
                                   "professor is hard to follow or reads from slides",
                                   "course pace is too fast or disorganized"],
    ("teaching", "negative"):     ["professor is hard to follow or reads from slides",
                                   "course pace is too fast or disorganized",
                                   "teaching excludes students from certain backgrounds"],
    ("coursecontent", "positive"): ["course content is relevant and up to date",
                                    "course content is rigorous and comprehensive"],
    ("coursecontent", "neutral"):  ["course content is relevant and up to date",
                                    "course content is outdated or misaligned with exams",
                                    "course content does not match what is tested on exams"],
    ("coursecontent", "negative"): ["course content is outdated or misaligned with exams",
                                    "course content does not match what is tested on exams",
                                    "course content assumes prior knowledge not listed"],
    ("examination", "positive"):  ["exam is fair and reflects what was taught"],
    ("examination", "neutral"):   ["exam is fair and reflects what was taught",
                                   "exam has bad weighting or no partial credit"],
    ("examination", "negative"):  ["exam is unfair or tests untaught material",
                                   "exam has bad weighting or no partial credit",
                                   "accommodation or accessibility failure in exam"],
    ("labwork", "positive"):      ["lab is hands-on and well organized"],
    ("labwork", "neutral"):       ["lab is hands-on and well organized",
                                   "lab equipment is broken or instructions are unclear"],
    ("labwork", "negative"):      ["lab equipment is broken or instructions are unclear",
                                   "lab is poorly organized or sessions are badly scheduled",
                                   "lab scheduling or location creates barriers"],
    ("library", "positive"):      ["generic positive feedback about the course"],
    ("library", "neutral"):       ["generic positive feedback about the course"],
    ("library", "negative"):      ["accommodation or accessibility failure in exam",
                                   "generic positive feedback about the course"],
    ("extracurricular", "positive"): ["generic positive feedback about the course"],
    ("extracurricular", "neutral"):  ["generic positive feedback about the course"],
    ("extracurricular", "negative"): ["generic positive feedback about the course"],
}


def _rating_bucket(r) -> str:
    try:
        v = float(r)
    except (TypeError, ValueError):
        return "neutral"
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def load_courseeval_labeled(
    filepath: str | Path | None = None,
    cfg: dict | None = None,
    cache_path: str | Path | None = None,
    zs_model: str = "facebook/bart-large-mnli",
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Load Purdue courseeval, run dimension-constrained zero-shot labeling,
    and return a labeled DataFrame ready to combine with RMP for training.

    Uses a narrow candidate set per (dimension, rating_bucket) so BART-MNLI
    picks from 2-3 relevant candidates instead of all 24, giving much higher accuracy.

    Returns DataFrame with 'text', 'sentiment', 'dimension', 'mode', 'source'.
    """
    import json
    import torch
    from transformers import pipeline as hf_pipeline

    df = load_courseeval(filepath)

    cfg = cfg or {}
    label_map = cfg.get("zero_shot_label_map", {}).get("professor", {})
    cache_file = Path(cache_path or (_PROJECT_ROOT / "zero_shot_cache_courseeval.json"))

    cached: dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        logger.info("[courseeval-zs] Loaded %d cached labels.", len(cached))

    device = 0 if torch.cuda.is_available() else -1
    zs = hf_pipeline("zero-shot-classification", model=zs_model, device=device)

    labels_out: list[str] = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        dim = str(row.get("dimension", "teaching")).lower()
        bucket = _rating_bucket(row.get("rating"))

        if text in cached:
            labels_out.append(cached[text])
            continue

        key = (dim, bucket)
        candidates = _DIMENSION_CANDIDATES.get(key, ["generic positive feedback about the course"])
        if len(candidates) == 1:
            # Only one option — no need to run model
            winner = label_map.get(candidates[0], "Majority_Positive")
        else:
            result = zs(text, candidates, multi_label=False)
            top = result["labels"][0]
            winner = label_map.get(top, "Majority_Positive")

        cached[text] = winner
        labels_out.append(winner)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f)

    df = df.copy()
    df["sentiment"] = labels_out
    df["source"] = "courseeval_labeled"
    logger.info(
        "[courseeval-zs] Labeled %d rows. Distribution: %s",
        len(df),
        df["sentiment"].value_counts().to_dict(),
    )
    return df


# ---------------------------------------------------------------------------
# Combined professor training set
# ---------------------------------------------------------------------------

def load_professor_combined(
    rmp_path: str | Path | None = None,
    courseeval_path: str | Path | None = None,
    rmp_n_per_class: int = 50_000,
    cfg: dict | None = None,
    include_courseeval_in_train: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df).

    train_df = RMP sample (full 24-class zero-shot) +
               courseeval labeled rows (dimension-constrained zero-shot)
               — training on both gives the model real university language.

    val_df   = held-out 20% of courseeval (never seen during training).

    include_courseeval_in_train=True is the default and recommended setting.
    Set to False to revert to the old RMP-only training.
    """
    from sklearn.model_selection import train_test_split

    rmp_df = load_rmp(rmp_path, n_per_class=rmp_n_per_class)

    ce_df = load_courseeval(courseeval_path)
    # Split courseeval: 80% train, 20% val
    ce_train, ce_val = train_test_split(ce_df, test_size=0.2, random_state=42)

    if include_courseeval_in_train:
        # Label the training split with dimension-constrained zero-shot
        ce_labeled = load_courseeval_labeled(courseeval_path, cfg=cfg)
        # Keep only the rows that ended up in ce_train (by index)
        ce_labeled_train = ce_labeled.loc[ce_labeled.index.isin(ce_train.index)].copy()
        train_df = pd.concat([rmp_df, ce_labeled_train], ignore_index=True)
        logger.info(
            "[professor-combined] Training set: %d RMP + %d courseeval = %d total",
            len(rmp_df), len(ce_labeled_train), len(train_df),
        )
    else:
        train_df = rmp_df

    return train_df, ce_val
