"""End-to-end inference pipeline: ingest → classify → minority detect → mismatch."""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_model(name: str | None = None) -> Path:
    """Locate a Hugging Face model directory on disk."""
    env = os.environ.get("FEEDBACK_MODEL_DIR", "").strip()
    raw = env or name or _cfg().get("model", {}).get("output_dir", "final_feedback_classifier")
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    for root in (_PROJECT_ROOT, Path.cwd()):
        cand = (root / p).resolve()
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand
    return (_PROJECT_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# CSV ingestion & column detection
# ---------------------------------------------------------------------------

# Feedback-like column name hints (ordered for scoring boost)
_TEXT_HINTS = (
    "feedback", "comment", "review", "text", "opinion", "response",
    "answer", "open", "written", "notes", "description",
    "teaching", "course", "exam", "lab", "peer", "team",
)


def _mean_text_len(series: pd.Series) -> float:
    s = series.dropna().astype(str)
    return float(s.str.len().mean()) if not s.empty else 0.0


def _is_string_like(df: pd.DataFrame, col: str) -> bool:
    dt = df[col].dtype
    return dt == object or pd.api.types.is_string_dtype(dt)


def auto_detect_text_columns(df: pd.DataFrame, max_cols: int = 12) -> list[str]:
    """Pick feedback-like text columns heuristically."""
    candidates: list[tuple[float, str]] = []
    for col in df.columns:
        if not _is_string_like(df, col):
            continue
        ml = _mean_text_len(df[col])
        if ml < 10.0:
            continue
        cl = col.lower().strip()
        if cl in ("id", "uuid") or cl.endswith("_id") or cl.startswith("unnamed"):
            continue
        score = ml + sum(80.0 for kw in _TEXT_HINTS if kw in cl)
        candidates.append((score, col))

    candidates.sort(key=lambda x: -x[0])
    out = [c for _, c in candidates[:max_cols]]

    if not out:  # Fallback: all non-id string columns
        out = [
            c for c in df.columns
            if _is_string_like(df, c)
            and c.lower().strip() not in ("id", "uuid")
            and not c.lower().strip().endswith("_id")
            and not c.lower().strip().startswith("unnamed")
        ][:max_cols]

    if not out:  # Last resort: longest columns by mean length
        scored = sorted(
            ((float(_mean_text_len(df[c].astype(str))), c) for c in df.columns),
            reverse=True,
        )
        out = [c for _, c in scored[:3]]

    return out


def _detect_rating_column(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    for col in df.columns:
        if col in text_cols or col == "text":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 10:
                return col
    return None


def _detect_label_column(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    """Check if a column looks like pre-existing sentiment labels."""
    for col in df.columns:
        if col in text_cols or col == "text":
            continue
        cl = col.lower()
        if cl in ("sentiment", "label", "class", "category"):
            return col
        if _is_string_like(df, col):
            unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
            standard_labels = {"positive", "negative", "neutral"}
            if unique_vals and unique_vals.issubset(standard_labels | {"0", "1", "2"}):
                return col
    return None


def _looks_like_headerless(first_row_value: str) -> bool:
    """Heuristic: if the first cell is a long sentence, the file has no header."""
    v = str(first_row_value).strip()
    return len(v) > 30 and " " in v


def ingest_csv(
    source: Union[str, Path, bytes, io.BytesIO],
    anonymize: bool = True,
    spacy_model: str = "en_core_web_sm",
) -> tuple[pd.DataFrame, list[str], str | None, str | None]:
    """
    Load a CSV, auto-detect columns, optionally anonymize.

    Returns:
        df            – DataFrame with a combined ``text`` column added.
        text_cols     – list of detected source text column names.
        rating_col    – numeric rating column name, or None.
        label_col     – pre-existing label column name, or None.
    """
    cfg = _cfg()
    placeholder = cfg.get("anonymization", {}).get("placeholder", "[STUDENT]")

    # ── Read file ──────────────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        raw_peek = pd.read_csv(source, header=None, nrows=2)
        is_headerless = _looks_like_headerless(raw_peek.iloc[0, 0])
        df = (
            pd.read_csv(source, header=None, names=["text"])
            if is_headerless
            else pd.read_csv(source)
        )
    else:
        if isinstance(source, bytes):
            source = io.BytesIO(source)
        content = source.read()
        raw_peek = pd.read_csv(io.BytesIO(content), header=None, nrows=2)
        is_headerless = _looks_like_headerless(raw_peek.iloc[0, 0])
        df = (
            pd.read_csv(io.BytesIO(content), header=None, names=["text"])
            if is_headerless
            else pd.read_csv(io.BytesIO(content))
        )

    if is_headerless:
        logger.info("[ingest] Headerless CSV detected → single 'text' column.")
        text_cols = ["text"]
    else:
        text_cols = auto_detect_text_columns(df)
        logger.info("[ingest] Detected text columns: %s", text_cols)

    # ── Rating & label columns ─────────────────────────────────────────────
    rating_col = _detect_rating_column(df, text_cols)
    label_col = _detect_label_column(df, text_cols)

    if rating_col:
        logger.info("[ingest] Detected rating column: %s", rating_col)
    if label_col:
        logger.info("[ingest] Detected label column: %s", label_col)

    # ── Combine text columns ───────────────────────────────────────────────
    if "text" not in df.columns:
        df["text"] = (
            df[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # ── Anonymize ─────────────────────────────────────────────────────────
    if anonymize:
        try:
            from .anonymizer import anonymize_series
            df["text"] = anonymize_series(df["text"], placeholder=placeholder, spacy_model=spacy_model)
            logger.info("[ingest] Anonymization applied.")
        except Exception as exc:
            logger.warning("[ingest] Anonymization failed (%s); continuing without.", exc)

    return df, text_cols, rating_col, label_col


# ---------------------------------------------------------------------------
# Label derivation
# ---------------------------------------------------------------------------

def derive_labels(
    df: pd.DataFrame,
    label_col: str | None,
    rating_col: str | None,
    use_zero_shot: bool = True,
    dataset_type: str = "auto",
) -> pd.DataFrame:
    """
    Add a ``sentiment`` column using the best available strategy:

    1. Use ``label_col`` directly if present.
    2. Derive from ``rating_col`` numeric thresholds.
    3. Zero-shot classify with BART-MNLI.

    Also determines ``dataset_type`` (``"course"`` or ``"peer"``) and adds
    expanded fine-grained labels when zero-shot is used.
    """
    out = df.copy()

    # Strategy 1 — explicit label column
    if label_col and label_col in out.columns:
        logger.info("[labels] Using explicit label column: %s", label_col)
        out["sentiment"] = out[label_col].astype(str).str.strip()
        return out

    # Strategy 2 — numeric rating
    if rating_col and rating_col in out.columns:
        logger.info("[labels] Deriving sentiment from rating column: %s", rating_col)
        ratings = pd.to_numeric(out[rating_col], errors="coerce")
        cfg_tr = _cfg().get("training", {})

        def _map(r):
            if pd.isna(r):
                return np.nan
            if r < 2.5:
                return "Negative"
            if r < 3.8:
                return "Neutral"
            return "Positive"

        out["sentiment"] = ratings.apply(_map)

        # Quantile-based fallback if only one class emerged
        if out["sentiment"].nunique(dropna=True) <= 1:
            logger.info("[labels] Rating thresholds produced one class → using quantiles.")
            q33 = ratings.quantile(0.33)
            q66 = ratings.quantile(0.66)
            out["sentiment"] = ratings.apply(
                lambda r: (
                    np.nan if pd.isna(r) else
                    "Negative" if r <= q33 else
                    "Neutral" if r <= q66 else
                    "Positive"
                )
            )
        return out

    # Strategy 3 — zero-shot BART-MNLI
    if use_zero_shot:
        logger.info("[labels] No labels/ratings found → zero-shot classification.")
        out = _zero_shot_label(out, dataset_type=dataset_type)
        return out

    raise ValueError(
        "Cannot derive labels: no label column, no rating column, and zero-shot is disabled."
    )


def _detect_dataset_type(texts: list[str]) -> str:
    """Heuristic: detect whether content is peer or course feedback."""
    sample = " ".join(texts[:200]).lower()
    peer_signals = ["teammate", "team member", "group member", "worked together", "contributed", "catme"]
    course_signals = ["professor", "lecture", "syllabus", "course material", "exam", "assignment"]
    peer_score = sum(s in sample for s in peer_signals)
    course_score = sum(s in sample for s in course_signals)
    return "peer" if peer_score >= course_score else "course"


def _zero_shot_label(
    df: pd.DataFrame,
    dataset_type: str = "auto",
    batch_size: int | None = None,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """
    Assign sentiment labels via facebook/bart-large-mnli.
    Results are cached to *cache_path* so interrupted runs can resume.
    """
    import json
    import torch
    from transformers import pipeline as hf_pipeline

    cfg_zs = _cfg().get("zero_shot", {})
    zs_model = cfg_zs.get("model", "facebook/bart-large-mnli")
    bs = batch_size or cfg_zs.get("batch_size", 32)
    cache_file = Path(cache_path or cfg_zs.get("cache_path", "zero_shot_labels_cache.json"))

    texts = df["text"].fillna("").astype(str).tolist()

    # Detect dataset type
    if dataset_type == "auto":
        dataset_type = _detect_dataset_type(texts)
    logger.info("[zero-shot] Dataset type detected: %s", dataset_type)

    # Load cache
    cached: dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        logger.info("[zero-shot] Loaded %d cached labels from %s", len(cached), cache_file)

    cfg_labels = _cfg().get("labels", {})
    candidate_labels = cfg_labels.get(dataset_type, cfg_labels.get("broad", ["Positive", "Neutral", "Negative"]))

    device = 0 if torch.cuda.is_available() else -1
    logger.info("[zero-shot] Running %s on %d texts (device=%s)…", zs_model, len(texts), "GPU" if device == 0 else "CPU")

    zs = hf_pipeline("zero-shot-classification", model=zs_model, device=device)

    to_label: list[int] = [i for i, t in enumerate(texts) if t not in cached]
    labels_out = [""] * len(texts)

    # Fill from cache first
    for i, t in enumerate(texts):
        if t in cached:
            labels_out[i] = cached[t]

    # Run zero-shot for uncached
    for start in range(0, len(to_label), bs):
        batch_idx = to_label[start : start + bs]
        batch_texts = [texts[i] for i in batch_idx]
        raw = zs(batch_texts, candidate_labels, multi_label=False)
        if isinstance(raw, dict):
            raw = [raw]
        for j, res in enumerate(raw):
            gi = batch_idx[j]
            winner = res["labels"][0]
            labels_out[gi] = winner
            cached[texts[gi]] = winner

        # Checkpoint cache every 500 rows
        if (start + bs) % 500 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            logger.info("[zero-shot] Checkpoint saved (%d/%d).", start + bs, len(to_label))

    # Final cache save
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f)

    out = df.copy()
    out["sentiment"] = labels_out
    out["dataset_type"] = dataset_type
    return out


# ---------------------------------------------------------------------------
# Inference (classifier)
# ---------------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    model_dir: str | Path | None = None,
    batch_size: int | None = None,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """Run the fine-tuned classifier on df['text'], adding prediction/confidence/needs_review."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = _cfg()
    bs = batch_size or cfg.get("model", {}).get("batch_size", 32)
    thresh = confidence_threshold or cfg.get("review_queue", {}).get("confidence_threshold", 0.65)

    mdir = _resolve_model(str(model_dir) if model_dir else None)
    if not (mdir.is_dir() and (mdir / "config.json").is_file()):
        raise FileNotFoundError(f"Model not found at {mdir}")

    tokenizer = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    texts = df["text"].fillna("").astype(str).tolist()
    predictions: list[str] = []
    confidences: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch = texts[start : start + bs]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1).cpu().tolist()
            conf = torch.max(probs, dim=1).values.cpu().tolist()
            predictions.extend([id2label[p] for p in pred_ids])
            confidences.extend(conf)

    out = df.copy()
    out["prediction"] = predictions
    out["confidence"] = [round(c, 4) for c in confidences]
    out["needs_review"] = [c < thresh for c in confidences]
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    source: Union[str, Path, bytes, io.BytesIO],
    model_dir: str | None = None,
    anonymize: bool = True,
    include_minority: bool = True,
    include_mismatch: bool = True,
    run_zero_shot_categorization: bool = False,
    confidence_threshold: float | None = None,
) -> pd.DataFrame:
    """
    One-shot: ingest → inference → minority detection → mismatch detection.

    Args:
        source:                    File path or raw bytes.
        model_dir:                 Path to fine-tuned classifier; auto-resolved if None.
        anonymize:                 Replace person names with [STUDENT].
        include_minority:          Run IsolationForest + DBSCAN minority detection.
        include_mismatch:          Compare predictions to numeric ratings.
        run_zero_shot_categorization: Use BART-MNLI for minority categorisation.
        confidence_threshold:      Below this → needs_review=True.
    """
    cfg = _cfg()

    # ── Ingest ────────────────────────────────────────────────────────────
    df, text_cols, rating_col, label_col = ingest_csv(
        source,
        anonymize=anonymize,
        spacy_model=cfg.get("anonymization", {}).get("spacy_model", "en_core_web_sm"),
    )
    df = df[df["text"].str.len() > 5].reset_index(drop=True)
    logger.info("[pipeline] After cleaning: %d rows.", len(df))

    # ── Auto-select model based on dataset type ───────────────────────────
    if model_dir is None:
        dataset_type = _detect_dataset_type(df["text"].dropna().tolist())
        logger.info("[pipeline] Detected dataset type: %s", dataset_type)
        if dataset_type == "peer":
            catme_dir = cfg.get("model", {}).get("catme_output_dir", "catme_feedback_classifier")
            catme_path = _resolve_model(catme_dir)
            if catme_path.is_dir() and (catme_path / "config.json").is_file():
                model_dir = str(catme_path)
                logger.info("[pipeline] Auto-selected CATME model: %s", model_dir)
            else:
                logger.warning("[pipeline] CATME model not found at %s; falling back to default.", catme_path)

    # ── Inference ─────────────────────────────────────────────────────────
    df = run_inference(df, model_dir=model_dir, confidence_threshold=confidence_threshold)

    # ── Minority detection ────────────────────────────────────────────────
    if include_minority:
        from .minority_detector import detect_minority_patterns

        md_cfg = cfg.get("minority_detection", {})
        emb_model = cfg.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
        df = detect_minority_patterns(
            df,
            df["text"].tolist(),
            embedding_model=emb_model,
            contamination=md_cfg.get("contamination", 0.08),
            n_estimators=md_cfg.get("n_estimators", 200),
            dbscan_eps=md_cfg.get("dbscan_eps", 0.75),
            dbscan_min_samples=md_cfg.get("dbscan_min_samples", 5),
            min_cluster_size=md_cfg.get("min_cluster_size", 10),
            include_dbscan_noise=md_cfg.get("include_dbscan_noise", False),
            pred_df=df,
            categorize=True,
            run_zero_shot_categorization=run_zero_shot_categorization,
        )

    # ── Mismatch detection ────────────────────────────────────────────────
    if include_mismatch and rating_col:
        from .mismatch_detector import detect_mismatches

        mm_cfg = cfg.get("mismatch", {})
        df = detect_mismatches(
            df,
            prediction_col="prediction",
            rating_col=rating_col,
            high_rating_threshold=mm_cfg.get("high_rating_threshold", 3.8),
            low_rating_threshold=mm_cfg.get("low_rating_threshold", 2.5),
        )

    return df
