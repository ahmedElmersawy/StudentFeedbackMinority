"""End-to-end inference pipeline: ingest → classify → minority detect → mismatch → priority."""
from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model cache — load each model once per process lifetime
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple] = {}  # str(path) -> (tokenizer, model, device, id2label)

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


def _resolve_model(name: Optional[str] = None) -> Path:
    """Locate a HuggingFace model directory on disk.

    Prefers /tmp/fa_models/<name> (fast local disk) over NFS home directory.
    On HPC clusters home directories are NFS-mounted — loading a 314 MB model
    from NFS can take 10+ minutes vs <10 seconds from local /tmp SSD.
    """
    if name is not None:
        raw = name
    else:
        env = os.environ.get("FEEDBACK_MODEL_DIR", "").strip()
        raw = env or _cfg().get("model", {}).get("catme_output_dir", "catme_feedback_classifier")

    p = Path(raw).expanduser()

    # Try local /tmp cache first (fast SSD, avoids NFS latency)
    if not p.is_absolute():
        local = Path("/tmp/fa_models") / p.name
        if local.is_dir() and (local / "config.json").is_file():
            return local.resolve()

    if p.is_absolute():
        return p.resolve()
    for root in (_PROJECT_ROOT, Path.cwd()):
        cand = (root / p).resolve()
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand
    return (_PROJECT_ROOT / p).resolve()


def _model_is_valid(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file()


# ---------------------------------------------------------------------------
# CATME subtype detection (Step 2)
# ---------------------------------------------------------------------------

_SELF_INDICATORS = [
    "i feel", "i did", "i worked", "i contributed",
    "i tried", "i have been", "i was able", "i am",
    "my contribution", "my work", "i could have",
    "i think i", "i believe i", "i slipped", "i fell",
    "i should have", "i plan to", "i will improve",
    "i let the team", "i wasnt", "i wasn't", "i have done",
    "i did my", "i completed my",
]


def detect_catme_subtype(text: str) -> str:
    """Returns 'self_assessment' or 'peer_feedback'."""
    tl = text.lower().strip()
    score = sum(1 for k in _SELF_INDICATORS if k in tl)
    if score >= 2:
        return "self_assessment"
    if score == 1 and len(text.split()) < 30:
        return "self_assessment"
    return "peer_feedback"


# ---------------------------------------------------------------------------
# Keyword-first minority detection (Step 3) — runs before embeddings
# ---------------------------------------------------------------------------

def keyword_minority_detection(text: str, keyword_map: dict[str, list[str]]) -> tuple[bool, list[str]]:
    """
    Returns (is_minority, matched_categories).
    Checks all keyword groups; collects all matches.
    Returns (False, ['Statistical_Outlier_Only']) if no match.
    """
    text_lower = text.lower()
    matched = [cat for cat, kws in keyword_map.items() if any(kw in text_lower for kw in kws)]
    if matched:
        return True, matched
    return False, ["Statistical_Outlier_Only"]


# ---------------------------------------------------------------------------
# Zero-shot classification (Step 5)
# ---------------------------------------------------------------------------

def zero_shot_classify(
    text: str,
    mode: str,
    subtype: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> tuple[str, float]:
    """
    Classify a single text using BART-MNLI.
    Returns (label, confidence).
    """
    import torch
    from transformers import pipeline as hf_pipeline

    cfg = cfg or _cfg()
    zs_model = cfg.get("zero_shot", {}).get("model", "facebook/bart-large-mnli")
    candidates_cfg = cfg.get("zero_shot_candidates", {})
    label_map_cfg = cfg.get("zero_shot_label_map", {})

    if mode == "student_to_student":
        key = subtype if subtype in candidates_cfg else "peer_feedback"
    else:
        key = "professor"

    candidates = candidates_cfg.get(key, ["Positive", "Neutral", "Negative"])
    label_map = label_map_cfg.get(key, {})

    device = 0 if torch.cuda.is_available() else -1
    zs = hf_pipeline("zero-shot-classification", model=zs_model, device=device)
    result = zs(text, candidates, multi_label=False)
    top_candidate = result["labels"][0]
    confidence = float(result["scores"][0])
    label = label_map.get(top_candidate, "Majority_Positive")
    return label, confidence


# ---------------------------------------------------------------------------
# Priority scoring (Step 7)
# ---------------------------------------------------------------------------

def calculate_priority(
    prediction: str,
    is_minority: bool,
    minority_categories: list[str],
    confidence: float,
    cfg: Optional[dict] = None,
) -> int:
    """
    Returns priority score 1–10.
    10 = urgent keyword-matched minority
    9  = high-priority label
    7  = medium-priority label
    5  = other
    1  = deprioritized (generic positive)
    +1 for low confidence (needs human review)
    """
    cfg = cfg or _cfg()
    priority_cfg = cfg.get("priority", {})
    high = set(priority_cfg.get("high", []))
    medium = set(priority_cfg.get("medium", []))
    deprioritized = set(priority_cfg.get("deprioritized", []))

    score: int
    if is_minority and "Statistical_Outlier_Only" not in minority_categories:
        score = 10
    elif prediction in high:
        score = 9
    elif prediction in medium:
        score = 7
    elif prediction in deprioritized:
        score = 1
    else:
        score = 5

    # Boost for low confidence — needs human review
    if confidence < 0.80:
        score = min(10, score + 1)

    return score


# ---------------------------------------------------------------------------
# CSV ingestion & column detection
# ---------------------------------------------------------------------------

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

    if not out:
        out = [
            c for c in df.columns
            if _is_string_like(df, c)
            and c.lower().strip() not in ("id", "uuid")
            and not c.lower().strip().endswith("_id")
            and not c.lower().strip().startswith("unnamed")
        ][:max_cols]

    if not out:
        scored = sorted(
            ((float(_mean_text_len(df[c].astype(str))), c) for c in df.columns),
            reverse=True,
        )
        out = [c for _, c in scored[:3]]

    return out


def _detect_rating_column(df: pd.DataFrame, text_cols: list[str]) -> Optional[str]:
    for col in df.columns:
        if col in text_cols or col == "text":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) > 0 and vals.min() >= 1 and vals.max() <= 10:
                return col
    return None


def _detect_label_column(df: pd.DataFrame, text_cols: list[str]) -> Optional[str]:
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
    v = str(first_row_value).strip()
    return len(v) > 30 and " " in v


def _melt_professor_dimensions(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Detect paired (rating_col, text_col) structures like studentdataset.csv and
    melt them into one row per dimension. Returns None if the pattern is not found.

    Detection: at least 2 numeric columns whose values are all in {-1, 0, 1} paired
    with adjacent text columns (col.1, Capitalized variant, or leading-space variant).
    """
    pairs: list[tuple[str, str, str]] = []  # (rating_col, text_col, dimension_name)
    checked: set[str] = set()

    col_list = list(df.columns)
    for col in col_list:
        if col in checked:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0 or not set(vals.unique()).issubset({-1.0, 0.0, 1.0}):
            continue

        # Look for a paired text column
        candidates = [
            col + ".1",
            col.capitalize(),
            " " + col,
            col + "_text",
            col + "_comments",
        ]
        for text_col in candidates:
            if text_col in df.columns and text_col not in checked:
                dim = col.strip().lower().replace(" ", "_")
                pairs.append((col, text_col, dim))
                checked.add(col)
                checked.add(text_col)
                break

    if len(pairs) < 2:
        return None

    rows: list[pd.DataFrame] = []
    for rating_col, text_col, dim in pairs:
        sub = df[[rating_col, text_col]].copy()
        sub.columns = ["rating", "text"]
        sub["dimension"] = dim
        sub = sub[sub["text"].notna() & (sub["text"].str.strip().str.len() > 5)]
        rows.append(sub)

    if not rows:
        return None

    result = pd.concat(rows, ignore_index=True)
    logger.info(
        "[pipeline] Courseeval-style CSV detected: %d pairs × %d source rows → %d melted rows",
        len(pairs), len(df), len(result),
    )
    return result


def ingest_csv(
    source: Union[str, Path, bytes, io.BytesIO],
    anonymize: bool = True,
    spacy_model: str = "en_core_web_sm",
) -> tuple[pd.DataFrame, list[str], Optional[str], Optional[str]]:
    """
    Load a CSV, auto-detect columns, optionally anonymize.

    Returns:
        df        – DataFrame with a combined ``text`` column added.
        text_cols – list of detected source text column names.
        rating_col – numeric rating column name, or None.
        label_col  – pre-existing label column name, or None.
    """
    cfg = _cfg()
    placeholder = cfg.get("anonymization", {}).get("placeholder", "[STUDENT]")

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

    rating_col = _detect_rating_column(df, text_cols)
    label_col = _detect_label_column(df, text_cols)

    if "text" not in df.columns:
        df["text"] = (
            df[text_cols]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    if anonymize:
        try:
            from .anonymizer import anonymize_series
            df["text"] = anonymize_series(df["text"], placeholder=placeholder, spacy_model=spacy_model)
            logger.info("[ingest] Anonymization applied.")
        except Exception as exc:
            logger.warning("[ingest] Anonymization failed (%s); continuing without.", exc)

    return df, text_cols, rating_col, label_col


# ---------------------------------------------------------------------------
# Feedback mode auto-detection
# ---------------------------------------------------------------------------

def detect_feedback_mode(texts: list[str]) -> str:
    """
    Returns 'student_to_student' or 'student_to_professor'.
    Heuristic based on vocabulary signals in the first 300 rows.
    """
    sample = " ".join(texts[:300]).lower()
    s2s_signals = ["teammate", "team member", "group member", "contributed", "catme", "worked together"]
    s2p_signals = ["professor", "lecture", "syllabus", "course material", "exam", "assignment"]
    s2s_score = sum(s in sample for s in s2s_signals)
    s2p_score = sum(s in sample for s in s2p_signals)
    return "student_to_student" if s2s_score >= s2p_score else "student_to_professor"


# ---------------------------------------------------------------------------
# Label derivation (legacy — still used for studentdataset.csv with ratings)
# ---------------------------------------------------------------------------

def derive_labels(
    df: pd.DataFrame,
    label_col: Optional[str],
    rating_col: Optional[str],
    use_zero_shot: bool = True,
    dataset_type: str = "auto",
) -> pd.DataFrame:
    out = df.copy()

    if label_col and label_col in out.columns:
        out["sentiment"] = out[label_col].astype(str).str.strip()
        return out

    if rating_col and rating_col in out.columns:
        ratings = pd.to_numeric(out[rating_col], errors="coerce")

        def _map(r):
            if pd.isna(r):
                return np.nan
            if r < 2.5:
                return "Negative"
            if r < 3.8:
                return "Neutral"
            return "Positive"

        out["sentiment"] = ratings.apply(_map)

        if out["sentiment"].nunique(dropna=True) <= 1:
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

    if use_zero_shot:
        out = _zero_shot_label(out, dataset_type=dataset_type)
        return out

    raise ValueError("Cannot derive labels: no label column, no rating column, and zero-shot is disabled.")


def _zero_shot_label(
    df: pd.DataFrame,
    dataset_type: str = "auto",
    batch_size: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    import json
    import torch
    from transformers import pipeline as hf_pipeline

    cfg_zs = _cfg().get("zero_shot", {})
    zs_model = cfg_zs.get("model", "facebook/bart-large-mnli")
    bs = batch_size or cfg_zs.get("batch_size", 32)
    cache_file = Path(cache_path or cfg_zs.get("cache_path", "zero_shot_labels_cache.json"))

    texts = df["text"].fillna("").astype(str).tolist()

    if dataset_type == "auto":
        dataset_type = detect_feedback_mode(texts)
    logger.info("[zero-shot] Dataset type: %s", dataset_type)

    cached: dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)

    cfg_labels = _cfg().get("labels", {})
    # Map mode → label key
    label_key = "peer" if "student" in dataset_type else dataset_type
    candidate_labels = cfg_labels.get(label_key, cfg_labels.get("broad", ["Positive", "Neutral", "Negative"]))

    device = 0 if torch.cuda.is_available() else -1
    zs = hf_pipeline("zero-shot-classification", model=zs_model, device=device)

    to_label: list[int] = [i for i, t in enumerate(texts) if t not in cached]
    labels_out = [""] * len(texts)
    for i, t in enumerate(texts):
        if t in cached:
            labels_out[i] = cached[t]

    for start in range(0, len(to_label), bs):
        batch_idx = to_label[start: start + bs]
        batch_texts = [texts[i] for i in batch_idx]
        raw = zs(batch_texts, candidate_labels, multi_label=False)
        if isinstance(raw, dict):
            raw = [raw]
        for j, res in enumerate(raw):
            gi = batch_idx[j]
            winner = res["labels"][0]
            labels_out[gi] = winner
            cached[texts[gi]] = winner

        if (start + bs) % 500 == 0:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f)

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
    model_dir: Union[str, Path, None] = None,
    batch_size: Optional[int] = None,
    confidence_threshold: Optional[float] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """Run the fine-tuned classifier on df['text']."""
    import os
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = _cfg()
    bs = batch_size or cfg.get("model", {}).get("batch_size", 32)
    thresh = confidence_threshold or cfg.get("review_queue", {}).get("confidence_threshold", 0.65)

    mdir = _resolve_model(str(model_dir) if model_dir else None)
    if not _model_is_valid(mdir):
        raise FileNotFoundError(f"Model not found at {mdir}")

    cache_key = str(mdir)
    if cache_key not in _model_cache:
        logger.info("[inference] Loading model from %s (first time — caching for reuse)", mdir)

        if progress_callback:
            progress_callback(0, 0, 0)   # keep SSE alive during load

        tokenizer = AutoTokenizer.from_pretrained(mdir)

        if progress_callback:
            progress_callback(0, 0, 0)   # tokenizer done — stay alive

        model = AutoModelForSequenceClassification.from_pretrained(mdir)
        model.eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                model = model.half().to("cuda")   # FP16 — 2× throughput on GPU
                device = "cuda"
                logger.info("[inference] GPU warmup (compiling CUDA kernels)…")
                if progress_callback:
                    progress_callback(0, 0, 0)   # keep SSE alive during warmup
                _dummy = tokenizer(["warmup"], return_tensors="pt", padding=True)
                _dummy = {k: v.to("cuda") for k, v in _dummy.items()}
                with torch.inference_mode():
                    model(**_dummy)
                torch.cuda.synchronize()
                logger.info("[inference] GPU warmup complete.")
            except torch.OutOfMemoryError:
                model = model.float().to("cpu")
                device = "cpu"
        else:
            device = "cpu"

        _model_cache[cache_key] = (tokenizer, model, device, id2label)
        logger.info("[inference] Model cached on %s (FP16=%s).", device, device == "cuda")
    else:
        logger.info("[inference] Using cached model from %s.", mdir)
        tokenizer, model, device, id2label = _model_cache[cache_key]

    # Auto-scale batch size: GPU can handle much larger batches than CPU
    if device == "cuda" and bs <= 64:
        bs = 512   # A30 has 24 GB — 512-row batches are safe for distilroberta
    elif device == "cuda" and bs < 256:
        bs = 256

    texts = df["text"].fillna("").astype(str).tolist()
    predictions: list[str] = []
    confidences: list[float] = []

    import time as _time
    t_start = _time.perf_counter()
    ctx = torch.inference_mode()   # slightly faster than no_grad, no autograd overhead

    with ctx:
        for start in range(0, len(texts), bs):
            batch = texts[start: start + bs]
            # Use dynamic padding (pad to longest in this batch, not global max_length)
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            try:
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                if device == "cuda":
                    # Keep on GPU until we need CPU arrays
                    probs = torch.softmax(logits.float(), dim=1)
                else:
                    probs = torch.softmax(logits, dim=1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                device = "cpu"
                model = model.float().to(device)
                enc = {k: v.to(device) for k, v in enc.items()}
                probs = torch.softmax(model(**enc).logits, dim=1)

            pred_ids = torch.argmax(probs, dim=1).cpu().tolist()
            conf     = torch.max(probs, dim=1).values.cpu().tolist()
            predictions.extend([id2label[p] for p in pred_ids])
            confidences.extend(conf)

            done = min(start + bs, len(texts))
            if progress_callback:
                elapsed = _time.perf_counter() - t_start
                speed   = done / elapsed if elapsed > 0 else 0
                progress_callback(done, len(texts), speed)

    out = df.copy()
    out["prediction"] = predictions
    out["confidence"] = [round(c, 4) for c in confidences]
    out["needs_review"] = [c < thresh for c in confidences]
    return out


# ---------------------------------------------------------------------------
# Mixed-signal detection
# ---------------------------------------------------------------------------

def _detect_mixed_signals(df: pd.DataFrame, cfg: Optional[dict] = None) -> pd.DataFrame:
    """
    Flag rows belonging to a professor/student who has BOTH positive and negative
    predictions across different feedback rows — the most actionable pattern.

    Example: Prof X gets Teaching_Positive_Clarity on one row but
    Exam_Negative_Unfair on another → every row for Prof X gains
    minority_category='Mixed_Signal_Pattern' and is_minority_pattern=True.

    Only negative/actionable rows for that entity are flagged (not the positive ones),
    so the result set stays focused on what needs to change.

    Falls back silently if no entity-ID column is found.
    """
    cfg = cfg or _cfg()
    md_cfg = cfg.get("minority_detection", {})
    entity_cols = md_cfg.get("entity_id_columns", [])

    group_col: Optional[str] = None
    for col in entity_cols:
        if col in df.columns:
            group_col = col
            break

    if group_col is None or "prediction" not in df.columns:
        return df

    high = set(cfg.get("priority", {}).get("high", []))
    medium = set(cfg.get("priority", {}).get("medium", []))
    deprioritized = set(cfg.get("priority", {}).get("deprioritized", []))
    negative_labels = high | medium

    mixed_entities: list = []
    for entity_id, group in df.groupby(group_col):
        preds = set(group["prediction"].dropna())
        if (preds & deprioritized) and (preds & negative_labels):
            mixed_entities.append(entity_id)

    if not mixed_entities:
        return df

    out = df.copy()
    entity_mask = out[group_col].isin(mixed_entities)
    # Only flag the negative/actionable rows for that entity, not the positive ones
    negative_mask = entity_mask & out["prediction"].isin(negative_labels)

    out.loc[negative_mask, "is_minority_pattern"] = True
    out.loc[negative_mask, "minority_category"] = out.loc[negative_mask, "minority_category"].apply(
        lambda c: "Mixed_Signal_Pattern"
        if (not c or c in ("", "Statistical_Outlier_Only"))
        else f"Mixed_Signal_Pattern|{c}"
    )
    logger.info(
        "[pipeline] Mixed-signal entities: %d → flagged %d negative rows.",
        len(mixed_entities), int(negative_mask.sum()),
    )
    return out


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(
    source: Union[str, Path, bytes, io.BytesIO],
    feedback_mode: Optional[str] = None,
    model_dir: Optional[str] = None,
    anonymize: bool = True,
    include_minority: bool = True,
    include_mismatch: bool = True,
    run_zero_shot_categorization: bool = False,
    confidence_threshold: Optional[float] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    One-shot: ingest → mode detection → inference → CATME subtype → keyword minority
    → embedding minority → priority scoring → mismatch detection.

    Args:
        source:                       File path or raw bytes.
        feedback_mode:                'student_to_student' | 'student_to_professor' | None (auto-detect).
        model_dir:                    Path to fine-tuned classifier; auto-resolved if None.
        anonymize:                    Replace person names with [STUDENT].
        include_minority:             Run IsolationForest + DBSCAN minority detection.
        include_mismatch:             Compare predictions to numeric ratings.
        run_zero_shot_categorization: Use BART-MNLI for minority categorisation.
        confidence_threshold:         Below this → needs_review=True.
    """
    cfg = _cfg()

    # Ingest
    df, text_cols, rating_col, label_col = ingest_csv(
        source,
        anonymize=anonymize,
        spacy_model=cfg.get("anonymization", {}).get("spacy_model", "en_core_web_sm"),
    )
    df = df[df["text"].str.len() > 5].reset_index(drop=True)
    logger.info("[pipeline] After cleaning: %d rows.", len(df))

    # Detect feedback mode if not provided
    if feedback_mode is None:
        feedback_mode = detect_feedback_mode(df["text"].dropna().tolist())
    logger.info("[pipeline] Feedback mode: %s", feedback_mode)

    # For professor mode: detect paired (rating, text) columns and melt into
    # one row per dimension. This preserves dimension context (teaching vs exam
    # vs lab) that would otherwise be lost by concatenating all text into one row.
    if feedback_mode == "student_to_professor":
        melted = _melt_professor_dimensions(df)
        if melted is not None:
            df = melted
            # Update rating_col to the standardised 'rating' column from the melt
            rating_col = "rating"

    df["feedback_mode"] = feedback_mode

    # CATME subtype detection
    if feedback_mode == "student_to_student":
        logger.info("[pipeline] Detecting CATME subtypes…")
        df["catme_subtype"] = df["text"].apply(detect_catme_subtype)
        counts = df["catme_subtype"].value_counts().to_dict()
        logger.info("[pipeline] Subtypes: %s", counts)

    # Auto-select model
    if model_dir is None:
        if feedback_mode == "student_to_student":
            catme_dir = cfg.get("model", {}).get("catme_output_dir", "catme_feedback_classifier")
            candidate = _resolve_model(catme_dir)
            if _model_is_valid(candidate):
                model_dir = str(candidate)
                logger.info("[pipeline] Using CATME model: %s", model_dir)
            else:
                logger.warning("[pipeline] CATME model not found at %s; trying default.", candidate)
        elif feedback_mode == "student_to_professor":
            prof_dir = cfg.get("model", {}).get("professor_output_dir", "professor_feedback_classifier")
            candidate = _resolve_model(prof_dir)
            if _model_is_valid(candidate):
                model_dir = str(candidate)
                logger.info("[pipeline] Using professor model: %s", model_dir)
            else:
                logger.warning("[pipeline] Professor model not found at %s; trying default.", candidate)

    # Inference
    df = run_inference(df, model_dir=model_dir, confidence_threshold=confidence_threshold,
                       progress_callback=progress_callback)

    # Keyword-first minority detection (Step 3 — before embeddings)
    # Vectorized: build a boolean mask per category using str.contains, then combine.
    keyword_map = cfg.get("minority_keywords", {})
    lower_texts = df["text"].str.lower().fillna("")
    cat_masks: dict[str, pd.Series] = {}
    for cat, keywords in keyword_map.items():
        pattern = "|".join(map(re.escape, keywords))
        cat_masks[cat] = lower_texts.str.contains(pattern, regex=True, na=False)

    kw_is_minority_series = pd.Series(False, index=df.index)
    kw_categories_series: list[str] = []
    for mask in cat_masks.values():
        kw_is_minority_series |= mask
    for i in df.index:
        cats = [cat for cat, mask in cat_masks.items() if mask.at[i]]
        kw_categories_series.append("|".join(cats))

    df["keyword_minority"] = kw_is_minority_series
    df["keyword_categories"] = kw_categories_series

    # Embedding-based minority detection (IsolationForest + DBSCAN)
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

        # Merge keyword + embedding minority flags
        # A row is minority if EITHER keyword OR embedding flags it
        df["is_minority_pattern"] = df["is_minority_pattern"] | df["keyword_minority"]
        # Prefer keyword categories when available
        df["minority_category"] = df.apply(
            lambda row: row["keyword_categories"] if row["keyword_minority"]
            else (row.get("minority_category", "") or ""),
            axis=1,
        )
    else:
        # Keyword-only minority detection
        df["is_minority_pattern"] = df["keyword_minority"]
        df["is_outlier"] = False
        df["is_minority_cluster"] = False
        df["is_noise"] = False
        df["cluster_id"] = -1
        df["minority_category"] = df["keyword_categories"]

    # Suppress positive-only statistical outliers (Step 3b)
    # IsolationForest fires on any unusual text, including unusually good comments.
    # Only keep Statistical_Outlier_Only flags when the prediction is negative/actionable.
    if cfg.get("minority_detection", {}).get("suppress_positive_outliers", True):
        deprioritized = set(cfg.get("priority", {}).get("deprioritized", []))
        suppressed = (
            ~df["keyword_minority"].astype(bool)
            & df["minority_category"].fillna("").str.contains("Statistical_Outlier_Only", na=False)
            & df["prediction"].isin(deprioritized)
        )
        df.loc[suppressed, "is_minority_pattern"] = False
        df.loc[suppressed, "minority_category"] = ""
        if suppressed.sum():
            logger.info("[pipeline] Suppressed %d positive-only statistical outliers.", suppressed.sum())

    # Mixed-signal detection (Step 3c)
    # When the same professor/student has both positive and negative predictions
    # across different feedback rows, flag those rows for action.
    if cfg.get("minority_detection", {}).get("mixed_signal_detection", True):
        df = _detect_mixed_signals(df, cfg)

    # Priority scoring (Step 7)
    df["priority_score"] = df.apply(
        lambda row: calculate_priority(
            prediction=str(row.get("prediction", "")),
            is_minority=bool(row.get("is_minority_pattern", False)),
            minority_categories=str(row.get("minority_category", "")).split("|"),
            confidence=float(row.get("confidence", 1.0)),
            cfg=cfg,
        ),
        axis=1,
    )

    # Mismatch detection
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

    # Sort by priority descending
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Output file generation
# ---------------------------------------------------------------------------

def aggregate_by_entity(
    df: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Collapse classified rows into one summary row per professor/dimension/entity.

    Resolution order for the grouping key:
      1. A column named in cfg.aggregation.entity_id_columns (professor_id, name…)
      2. 'dimension' column — present when studentdataset.csv is melted
      3. 'feedback_mode' — fallback (single aggregate for the whole dataset)

    Returns one row per entity with:
      - total_responses, label_counts (dict), top_negative_label,
        minority_count, avg_priority_score,
        flagged_findings (labels that meet the threshold),
        needs_attention (bool)
    """
    cfg = cfg or _cfg()
    agg_cfg = cfg.get("aggregation", {})
    threshold = int(agg_cfg.get("min_students_threshold", 3))
    min_pct = float(agg_cfg.get("min_pct_threshold", 10.0))
    entity_cols = agg_cfg.get("entity_id_columns", [])

    high = set(cfg.get("priority", {}).get("high", []))
    medium = set(cfg.get("priority", {}).get("medium", []))
    actionable = high | medium

    # Choose grouping column
    group_col: str = "feedback_mode"
    for col in entity_cols:
        if col in df.columns:
            group_col = col
            break
    if group_col == "feedback_mode" and "dimension" in df.columns:
        group_col = "dimension"

    logger.info("[aggregate] Grouping by '%s'", group_col)

    rows: list[dict] = []
    for entity_id, group in df.groupby(group_col):
        total = len(group)
        if total == 0:
            continue

        label_counts: dict[str, int] = {}
        if "prediction" in group.columns:
            label_counts = group["prediction"].value_counts().to_dict()

        # Flagged findings: actionable labels with enough responses
        findings: list[dict] = []
        for label, count in label_counts.items():
            pct = 100.0 * count / total
            if label in actionable and count >= threshold and pct >= min_pct:
                findings.append({"label": label, "count": count, "pct": round(pct, 1)})
        findings.sort(key=lambda x: -x["count"])

        minority_count = int(df.get("is_minority_pattern", pd.Series(dtype=bool)).reindex(group.index).fillna(False).sum()) if "is_minority_pattern" in group.columns else 0
        avg_priority = float(group["priority_score"].mean()) if "priority_score" in group.columns else 0.0

        top_negative = next(
            (f["label"] for f in findings if f["label"] in high), None
        ) or next(
            (f["label"] for f in findings if f["label"] in medium), None
        )

        rows.append({
            "entity": str(entity_id),
            "group_by": group_col,
            "total_responses": total,
            "label_counts": label_counts,
            "top_negative_label": top_negative or "",
            "flagged_findings": findings,
            "needs_attention": len(findings) > 0,
            "minority_count": minority_count,
            "avg_priority_score": round(avg_priority, 2),
        })

    rows.sort(key=lambda r: (-len(r["flagged_findings"]), -r["avg_priority_score"]))
    return pd.DataFrame(rows)


def generate_output_files(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    feedback_mode: str,
) -> dict[str, Path]:
    """
    Write split output CSVs per spec.

    Student→Student:
      peer_feedback_results.csv
      self_assessment_results.csv
      priority_alerts.csv      (top 100 by priority_score)
      minority_experiences.csv (real minority, not Statistical_Outlier_Only)
      negative_peer_only.csv   (only Negative_ labels)

    Student→Professor:
      teaching_results.csv
      content_results.csv
      exam_results.csv
      lab_results.csv
      support_results.csv
      priority_alerts.csv
      minority_experiences.csv

    Returns dict of {name: path} for all files written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def _save(name: str, subset: pd.DataFrame) -> Path:
        p = out_dir / name
        subset.to_csv(p, index=False)
        logger.info("[output] %s → %d rows", name, len(subset))
        written[name] = p
        return p

    # Shared outputs
    top100 = df.nlargest(100, "priority_score")
    _save("priority_alerts.csv", top100)

    real_minority = df[
        df["is_minority_pattern"].astype(bool) &
        ~df["minority_category"].fillna("").str.contains("Statistical_Outlier_Only")
    ]
    _save("minority_experiences.csv", real_minority)

    if feedback_mode == "student_to_student":
        if "catme_subtype" in df.columns:
            peer = df[df["catme_subtype"] == "peer_feedback"]
            self_rows = df[df["catme_subtype"] == "self_assessment"]
        else:
            peer = df
            self_rows = pd.DataFrame(columns=df.columns)

        _save("peer_feedback_results.csv", peer)
        _save("self_assessment_results.csv", self_rows)

        negative_peer = peer[peer["prediction"].str.startswith("Negative_", na=False)]
        _save("negative_peer_only.csv", negative_peer)

    elif feedback_mode == "student_to_professor":
        _save("teaching_results.csv", df[df["prediction"].str.startswith("Teaching_", na=False)])
        _save("content_results.csv", df[df["prediction"].str.startswith("Content_", na=False)])
        _save("exam_results.csv", df[df["prediction"].str.startswith("Exam_", na=False)])
        _save("lab_results.csv", df[df["prediction"].str.startswith("Lab_", na=False)])
        _save("support_results.csv", df[df["prediction"].str.startswith("Support_", na=False)])

        # Dimension-level summary — the most actionable output for administrators
        summary_df = aggregate_by_entity(df)
        if not summary_df.empty:
            # Flatten flagged_findings list to a readable string for CSV
            summary_df["flagged_findings_str"] = summary_df["flagged_findings"].apply(
                lambda fs: "; ".join(f"{f['label']} ({f['count']} students, {f['pct']}%)" for f in fs)
            )
            summary_df["label_counts_str"] = summary_df["label_counts"].apply(
                lambda d: "; ".join(f"{k}: {v}" for k, v in sorted(d.items(), key=lambda x: -x[1])[:8])
            )
            cols = ["entity", "total_responses", "needs_attention",
                    "top_negative_label", "flagged_findings_str",
                    "minority_count", "avg_priority_score", "label_counts_str"]
            _save("dimension_summary.csv", summary_df[[c for c in cols if c in summary_df.columns]])

    return written


# ---------------------------------------------------------------------------
# Success criteria validation
# ---------------------------------------------------------------------------

def check_success_criteria(df: pd.DataFrame, feedback_mode: str) -> dict:
    """
    Return a dict of pass/fail checks per spec success criteria.
    Used by the API to surface warnings to the user.
    """
    results: dict[str, bool | int | str] = {}
    total = len(df)

    if feedback_mode == "student_to_student":
        majority_pct = (df["prediction"] == "Majority_Positive").mean() * 100
        negative_count = df["prediction"].str.startswith("Negative_", na=False).sum()
        real_minority = (
            df["is_minority_pattern"].astype(bool) &
            ~df["minority_category"].fillna("").str.contains("Statistical_Outlier_Only")
        ).sum()
        results["majority_positive_pct"] = round(float(majority_pct), 1)
        results["negative_labeled_count"] = int(negative_count)
        results["real_minority_count"] = int(real_minority)
        results["pass_majority_pct"] = majority_pct < 30
        results["pass_negative_count"] = negative_count >= 3000
        results["pass_minority_count"] = real_minority >= 500
        results["no_crash"] = True

    elif feedback_mode == "student_to_professor":
        majority_pct = df["prediction"].isin(["Majority_Positive", "Teaching_Positive_Clarity",
                                               "Teaching_Positive_Engagement"]).mean() * 100
        n_categories = df["prediction"].nunique()
        real_minority = (
            df["is_minority_pattern"].astype(bool) &
            ~df["minority_category"].fillna("").str.contains("Statistical_Outlier_Only")
        ).sum()
        results["majority_positive_pct"] = round(float(majority_pct), 1)
        results["n_label_categories"] = int(n_categories)
        results["real_minority_count"] = int(real_minority)
        results["pass_majority_pct"] = majority_pct < 30
        results["pass_label_spread"] = n_categories >= 8
        results["pass_minority_count"] = real_minority > 0

    return results
