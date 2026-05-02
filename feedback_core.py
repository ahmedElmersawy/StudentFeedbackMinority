"""Shared ML helpers for Streamlit and Reflex frontends (no UI imports)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def normalize_id2label(raw) -> dict[int, str]:
    if raw is None:
        return {}
    out: dict[int, str] = {}
    for k, v in raw.items():
        out[int(k)] = str(v)
    return out


def build_metadata(model_path: Path, model_config) -> dict:
    merged: dict = {
        "text_cols": [],
        "embedding_model": "all-MiniLM-L6-v2",
        "contamination": 0.08,
        "min_cluster_size": 10,
    }
    mf = model_path / "metadata.json"
    lf = model_path / "label_mappings.json"
    if mf.exists():
        merged.update(json.loads(mf.read_text(encoding="utf-8")))
    if lf.exists():
        lm = json.loads(lf.read_text(encoding="utf-8"))
        merged.setdefault("label2id", lm.get("label2id", {}))
        merged.setdefault("id2label", lm.get("id2label", {}))
    cfg_map = getattr(model_config, "id2label", None)
    if cfg_map:
        merged["id2label"] = normalize_id2label(cfg_map)
    return merged


def guess_segment_columns(df: pd.DataFrame, max_unique: int = 45, min_unique: int = 2) -> list[str]:
    out: list[str] = []
    for c in df.columns:
        if df[c].dtype == "object" or str(df[c].dtype).startswith(("category", "string")):
            n = int(df[c].nunique(dropna=True))
            if min_unique <= n <= max_unique:
                out.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            n = int(df[c].nunique(dropna=True))
            if min_unique <= n <= max_unique:
                out.append(c)
    return sorted(out, key=lambda col: df[col].nunique(dropna=True))


def aligned_prediction_counts(s1: pd.Series, s2: pd.Series) -> tuple[pd.Series, pd.Series]:
    labels = sorted(set(s1.index.astype(str)) | set(s2.index.astype(str)))
    a = s1.reindex(labels, fill_value=0)
    b = s2.reindex(labels, fill_value=0)
    return a, b


def combined_text_series(df: pd.DataFrame, selected_text_cols: list[str]) -> pd.Series:
    return (
        df[selected_text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_classifier(model_path: Path):
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {model_path}")
    if not list(model_path.glob("config.json")):
        raise FileNotFoundError("No config.json — not a Transformers model folder.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    metadata = build_metadata(model_path, model.config)
    if not metadata.get("id2label"):
        raise ValueError("Could not read id2label from model config or label_mappings.json.")
    return tokenizer, model, metadata


def resolve_classifier_dir(user_input: str | None = None) -> Path:
    """Resolve the Hugging Face classifier folder on disk.

    Precedence: ``FEEDBACK_MODEL_DIR`` environment variable, else *user_input*,
    else ``final_feedback_classifier``. Relative paths are tried under the repo
    root (directory containing this file) and under the process current working
    directory, so local and Streamlit Cloud layouts both work when the folder
    is committed next to ``app.py``.
    """
    env = os.environ.get("FEEDBACK_MODEL_DIR", "").strip()
    raw = env or (user_input or "").strip() or "final_feedback_classifier"
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    repo_root = Path(__file__).resolve().parent
    for root in (repo_root, Path.cwd()):
        cand = (root / p).resolve()
        if cand.is_dir() and (cand / "config.json").is_file():
            return cand
    return (repo_root / p).resolve()


def predict_dataframe(
    df: pd.DataFrame,
    combined_text: pd.Series,
    tokenizer,
    model,
    id2label: dict[int, str],
    batch_size: int = 32,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    predictions: list[str] = []
    confidences: list[float] = []
    texts = combined_text.tolist()
    id2label_int = {int(k): v for k, v in id2label.items()}
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1).cpu().tolist()
            pred_conf = torch.max(probs, dim=1).values.cpu().tolist()
            predictions.extend([id2label_int[p] for p in pred_ids])
            confidences.extend(pred_conf)
    out = df.copy()
    out["prediction"] = predictions
    out["confidence"] = confidences
    return out


def detect_minority_patterns(
    df: pd.DataFrame,
    texts: list[str],
    embedding_model: str,
    contamination: float,
    min_cluster_size: int,
    include_dbscan_noise: bool,
    pred_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    outlier_labels = iso.fit_predict(embeddings)
    outlier_mask = outlier_labels == -1

    clusterer = DBSCAN(eps=0.75, min_samples=5, metric="euclidean")
    cluster_labels = clusterer.fit_predict(embeddings)

    minority_cluster_mask = np.zeros(len(texts), dtype=bool)
    for cid in np.unique(cluster_labels):
        if cid == -1:
            continue
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) < min_cluster_size:
            minority_cluster_mask[idx] = True

    noise_mask = cluster_labels == -1
    minority_mask = outlier_mask | minority_cluster_mask
    if include_dbscan_noise:
        minority_mask |= noise_mask

    minority_df = df.copy()
    minority_df["is_outlier"] = outlier_mask
    minority_df["is_minority_cluster"] = minority_cluster_mask
    minority_df["is_noise"] = noise_mask
    minority_df["cluster_id"] = cluster_labels
    minority_df["is_minority_pattern"] = minority_mask
    if pred_df is not None:
        minority_df["prediction"] = pred_df["prediction"].values
        minority_df["confidence"] = pred_df["confidence"].values
    return minority_df


def default_text_columns(df: pd.DataFrame, metadata_text_cols: list[str]) -> list[str]:
    available = df.columns.tolist()
    suggested = [c for c in (metadata_text_cols or []) if c in available]
    if suggested:
        return suggested
    return [c for c in available if df[c].dtype == "object"][:4]


_FEEDBACK_NAME_HINTS = (
    "feedback",
    "comment",
    "review",
    "text",
    "opinion",
    "response",
    "answer",
    "open",
    "written",
    "notes",
    "description",
    "teaching",
    "course",
    "exam",
    "lab",
    "library",
    "extra",
    "curricular",
)


def _mean_text_len(series: pd.Series) -> float:
    s = series.dropna().astype(str)
    if s.empty:
        return 0.0
    return float(s.str.len().mean())


def auto_detect_text_columns(
    df: pd.DataFrame,
    metadata_text_cols: list[str] | None = None,
    max_cols: int = 12,
) -> list[str]:
    """Pick feedback-like text columns without user input (metadata wins when present)."""
    available = df.columns.tolist()
    meta = [c for c in (metadata_text_cols or []) if c in available]
    if meta:
        return meta[:max_cols]

    def is_string_like(col: str) -> bool:
        dt = df[col].dtype
        return dt == object or str(dt).startswith(("string", "category"))

    candidates: list[tuple[float, str]] = []
    for c in available:
        if not is_string_like(c):
            continue
        mlen = _mean_text_len(df[c])
        if mlen < 10.0:
            continue
        cl = c.lower().strip()
        if cl in ("id", "uuid") or cl.endswith("_id") or cl.startswith("unnamed"):
            continue
        score = mlen
        for kw in _FEEDBACK_NAME_HINTS:
            if kw in cl:
                score += 80.0
                break
        candidates.append((score, c))

    candidates.sort(key=lambda x: -x[0])
    out = [c for _, c in candidates[:max_cols]]
    if not out:
        # Fallback 1: keep string-like columns except obvious identifiers.
        objs = []
        for c in available:
            if not is_string_like(c):
                continue
            cl = c.lower().strip()
            if cl in ("id", "uuid") or cl.endswith("_id") or cl.startswith("unnamed"):
                continue
            objs.append(c)
        out = objs[: min(6, max_cols)]

    if not out:
        # Fallback 2: last resort for datasets that were typed as numeric/categorical
        # despite being semantically text-like after CSV parsing.
        broad: list[tuple[float, str]] = []
        for c in available:
            cl = c.lower().strip()
            if cl in ("id", "uuid") or cl.endswith("_id") or cl.startswith("unnamed"):
                continue
            mlen = _mean_text_len(df[c].astype(str))
            broad.append((mlen, c))
        broad.sort(key=lambda x: -x[0])
        out = [c for _, c in broad[: min(3, max_cols)]]
    return out


def auto_pick_segment_column(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    """First low-cardinality column not used as combined text, or None."""
    avoid = set(text_cols)
    for col in guess_segment_columns(df):
        if col not in avoid:
            return col
    return None


def counts_to_chart_rows(counts: pd.Series) -> list[dict]:
    return [{"label": str(i), "count": int(v)} for i, v in counts.items()]

