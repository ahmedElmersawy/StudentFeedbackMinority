"""
Minority-pattern detection and experiential-category classification.

DO NOT CHANGE:
  - all-MiniLM-L6-v2 embeddings
  - IsolationForest (global outliers) + DBSCAN (small clusters)
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# One GPU job at a time — prevents concurrent SentenceTransformer loads from OOMing
_gpu_lock = threading.Semaphore(1)

# Global SentenceTransformer cache — load once, reuse across all jobs
_embedder_cache: dict[str, object] = {}

_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ---------------------------------------------------------------------------
# Config — cached to avoid repeated YAML I/O on every detection call
# ---------------------------------------------------------------------------

_cfg_cache: Optional[dict] = None

def _cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _cfg_cache = yaml.safe_load(f) or {}
        else:
            _cfg_cache = {}
    return _cfg_cache


# ---------------------------------------------------------------------------
# Keyword-based first pass
# ---------------------------------------------------------------------------

def _keyword_match(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    """Return all categories whose keyword list has a match in *text*."""
    tl = text.lower()
    return [cat for cat, kws in keyword_map.items() if any(kw in tl for kw in kws)]


# ---------------------------------------------------------------------------
# Core detection (DO NOT CHANGE algorithms)
# ---------------------------------------------------------------------------

def detect_minority_patterns(
    df: pd.DataFrame,
    texts: list[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    contamination: float = 0.08,
    n_estimators: int = 200,
    dbscan_eps: float = 0.75,
    dbscan_min_samples: int = 5,
    min_cluster_size: int = 10,
    include_dbscan_noise: bool = False,
    pred_df: Optional[pd.DataFrame] = None,
    categorize: bool = True,
    run_zero_shot_categorization: bool = False,
) -> pd.DataFrame:
    """
    Two-stage minority detection:
      1. IsolationForest — global statistical outliers in embedding space.
      2. DBSCAN — small clusters (tightly-grouped rare patterns).

    After flagging, rows are categorised into experiential groups via keyword
    matching and optional zero-shot classification.

    Keyword categories include:
      - Minority experience categories (International_Student, First_Generation, etc.)
      - Negative_Peer_Flag (unresponsive, did no work, ghosted)
      - Suggestion_Flag (constructive improvement suggestions)
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import IsolationForest

    cfg_md = _cfg().get("minority_detection", {})
    contamination = contamination or cfg_md.get("contamination", 0.08)
    n_estimators = n_estimators or cfg_md.get("n_estimators", 200)
    dbscan_eps = dbscan_eps or cfg_md.get("dbscan_eps", 0.75)
    dbscan_min_samples = dbscan_min_samples or cfg_md.get("dbscan_min_samples", 5)
    min_cluster_size = min_cluster_size or cfg_md.get("min_cluster_size", 10)
    include_dbscan_noise = include_dbscan_noise or cfg_md.get("include_dbscan_noise", False)

    # For very large datasets, sample texts for minority detection to avoid OOM.
    # IsolationForest + DBSCAN on 816k × 384-dim exhausts system RAM.
    # We run detection on a representative sample, then assign labels via
    # nearest-neighbour (same approach as DBSCAN sampling already does).
    EMB_SAMPLE_CAP = int(cfg_md.get("emb_sample_cap", 100_000))
    full_texts = texts
    full_n = len(texts)
    sampled = False
    sample_idx: Optional[np.ndarray] = None

    if full_n > EMB_SAMPLE_CAP:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(full_n, EMB_SAMPLE_CAP, replace=False)
        texts = [full_texts[i] for i in sample_idx]
        logger.info("Large dataset (%d rows): capping minority detection at %d sampled rows.", full_n, EMB_SAMPLE_CAP)
        sampled = True

    # Deduplicate: encode unique texts only, then map back to full list.
    unique_texts = list(dict.fromkeys(texts))
    import hashlib, os, torch as _torch
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _device = "cuda" if _torch.cuda.is_available() else "cpu"

    # ── Persistent embedding cache ─────────────────────────────────────────
    _CACHE_DIR = Path("/tmp/fa_emb_cache")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _text_hash(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8", errors="replace")).hexdigest()[:16]

    cached_embs:   dict[str, np.ndarray] = {}
    uncached_texts: list[str] = []
    for t in unique_texts:
        h   = _text_hash(t)
        fp  = _CACHE_DIR / f"{h}.npy"
        if fp.exists():
            try:
                cached_embs[t] = np.load(fp)
            except Exception:
                uncached_texts.append(t)
        else:
            uncached_texts.append(t)

    logger.info("Embedding %d unique texts (%d total, %d cached) on %s…",
                len(unique_texts), len(texts), len(cached_embs), _device)

    if uncached_texts:
        # Acquire GPU lock — prevents concurrent jobs from OOMing the GPU
        with _gpu_lock:
            if embedding_model not in _embedder_cache:
                _embedder_cache[embedding_model] = SentenceTransformer(embedding_model, device=_device)
            embedder = _embedder_cache[embedding_model]
            _emb_bs  = 2048 if _device == "cuda" else 512
            new_embs = embedder.encode(
                uncached_texts, batch_size=_emb_bs, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=False,
            )
        for t, vec in zip(uncached_texts, new_embs):
            h  = _text_hash(t)
            fp = _CACHE_DIR / f"{h}.npy"
            try:
                np.save(fp, vec)
            except Exception:
                pass
            cached_embs[t] = vec

    # Reconstruct full embedding matrix in original order
    unique_emb = np.stack([cached_embs[t] for t in unique_texts])
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    emb = unique_emb[[text_to_idx[t] for t in texts]]

    # IsolationForest (DO NOT CHANGE)
    iso = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=42, n_jobs=-1)
    outlier_labels = iso.fit_predict(emb)
    outlier_mask = outlier_labels == -1

    # DBSCAN small clusters (DO NOT CHANGE algorithm).
    # For large datasets run DBSCAN on a random sample and assign labels to the
    # rest via nearest-neighbour lookup — turns O(n²) into O(sample²).
    DBSCAN_SAMPLE = 2000
    n_emb = len(texts)  # may be capped by EMB_SAMPLE_CAP
    if n_emb > DBSCAN_SAMPLE:
        logger.info("Large dataset (%d rows): DBSCAN on %d-row sample + GPU nearest-neighbor assignment.",
                    n_emb, DBSCAN_SAMPLE)
        rng_db = np.random.default_rng(43)
        dbscan_idx = rng_db.choice(n_emb, DBSCAN_SAMPLE, replace=False)
        dbscan_emb = emb[dbscan_idx]

        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean")
        dbscan_labels = clusterer.fit_predict(dbscan_emb)

        try:
            import torch as _torch
            if _torch.cuda.is_available():
                logger.info("[minority] GPU nearest-neighbor assignment…")
                _sample_t = _torch.from_numpy(dbscan_emb).cuda()
                _GPU_CHUNK = 10_000
                nn_idx_parts = []
                for start in range(0, n_emb, _GPU_CHUNK):
                    _chunk = _torch.from_numpy(emb[start:start + _GPU_CHUNK]).cuda()
                    _dists = _torch.cdist(_chunk, _sample_t)
                    nn_idx_parts.append(_torch.argmin(_dists, dim=1).cpu().numpy())
                nn_idx = np.concatenate(nn_idx_parts)
            else:
                raise RuntimeError("no cuda")
        except Exception as _e:
            logger.info("[minority] CPU nearest-neighbor fallback (%s).", _e)
            from sklearn.neighbors import NearestNeighbors
            _nn = NearestNeighbors(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=-1)
            _nn.fit(dbscan_emb)
            nn_idx_parts = []
            for start in range(0, n_emb, 50_000):
                _, _idx = _nn.kneighbors(emb[start:start + 50_000])
                nn_idx_parts.append(_idx.flatten())
            nn_idx = np.concatenate(nn_idx_parts)

        cluster_labels = dbscan_labels[nn_idx]
    else:
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric="euclidean")
        cluster_labels = clusterer.fit_predict(emb)

    minority_cluster_mask = np.zeros(n_emb, dtype=bool)
    for cid in np.unique(cluster_labels):
        if cid == -1:
            continue
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) < min_cluster_size:
            minority_cluster_mask[idx] = True

    noise_mask = cluster_labels == -1
    minority_mask: np.ndarray = outlier_mask | minority_cluster_mask
    if include_dbscan_noise:
        minority_mask |= noise_mask

    out = df.copy()
    if sampled and sample_idx is not None:
        # Map sample-level flags back to full DataFrame (unsampled rows → False / -1)
        full_is_outlier  = np.zeros(full_n, dtype=bool)
        full_is_cluster  = np.zeros(full_n, dtype=bool)
        full_is_noise    = np.zeros(full_n, dtype=bool)
        full_is_minority = np.zeros(full_n, dtype=bool)
        full_cluster_ids = np.full(full_n, -1, dtype=int)
        full_is_outlier[sample_idx]  = outlier_mask
        full_is_cluster[sample_idx]  = minority_cluster_mask
        full_is_noise[sample_idx]    = noise_mask
        full_is_minority[sample_idx] = minority_mask
        full_cluster_ids[sample_idx] = cluster_labels
        out["is_outlier"]          = full_is_outlier
        out["is_minority_cluster"] = full_is_cluster
        out["is_noise"]            = full_is_noise
        out["cluster_id"]          = full_cluster_ids
        out["is_minority_pattern"] = full_is_minority
        # Use full_texts for categorisation
        texts = full_texts
        minority_mask = full_is_minority
    else:
        out["is_outlier"]          = outlier_mask
        out["is_minority_cluster"] = minority_cluster_mask
        out["is_noise"]            = noise_mask
        out["cluster_id"]          = cluster_labels
        out["is_minority_pattern"] = minority_mask

    if pred_df is not None and "prediction" in pred_df.columns:
        out["prediction"] = pred_df["prediction"].values
        out["confidence"] = pred_df["confidence"].values

    if categorize:
        keyword_map = _cfg().get("minority_keywords", {})
        zs_model = _cfg().get("zero_shot", {}).get("model") if run_zero_shot_categorization else None
        cats = categorize_minority_rows(
            texts, minority_mask, keyword_map=keyword_map, zero_shot_model=zs_model
        )
        out["minority_category"] = cats

    n_min = int(minority_mask.sum())
    logger.info(
        "Minority detection complete: %d rows flagged (%.1f%%)",
        n_min,
        100.0 * n_min / max(1, len(texts)),
    )
    return out


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

def categorize_minority_rows(
    texts: list[str],
    minority_mask: "np.ndarray",
    keyword_map: Optional[dict[str, list[str]]] = None,
    zero_shot_model: Optional[str] = None,
    zero_shot_batch_size: int = 32,
) -> list[str]:
    """
    Assign each flagged row to one or more experiential categories.

    Pass 1: keyword matching (fast) — covers all categories including
            Negative_Peer_Flag and Suggestion_Flag.
    Pass 2: zero-shot classification for rows Pass 1 missed (optional).

    Returns a list of ``"|"``-joined category strings (empty for non-flagged rows).
    """
    cfg = _cfg()
    if keyword_map is None:
        keyword_map = cfg.get("minority_keywords", {})

    all_categories: list[str] = cfg.get("minority_categories", list(keyword_map.keys()))
    # Zero-shot only for actual minority experience categories (not Negative_Peer_Flag / Suggestion_Flag)
    zs_candidates = [
        c for c in all_categories
        if c not in ("Statistical_Outlier_Only", "Negative_Peer_Flag", "Suggestion_Flag")
    ]

    results: list[str] = [""] * len(texts)
    needs_zs: list[int] = []

    # Vectorized keyword matching on flagged texts only
    flagged_indices = [i for i, v in enumerate(minority_mask) if v]
    if flagged_indices and keyword_map:
        import re as _re
        flagged_texts_lower = [texts[i].lower() for i in flagged_indices]
        cat_names = list(keyword_map.keys())
        # Build regex per category: compile once, apply to all texts at once
        cat_patterns = {
            cat: _re.compile("|".join(_re.escape(kw) for kw in kws), _re.IGNORECASE)
            for cat, kws in keyword_map.items() if kws
        }
        for j, (gi, tl) in enumerate(zip(flagged_indices, flagged_texts_lower)):
            matched = [cat for cat, pat in cat_patterns.items() if pat.search(tl)]
            if matched:
                results[gi] = "|".join(matched)
            else:
                needs_zs.append(gi)
    elif flagged_indices:
        needs_zs = flagged_indices

    # Optional zero-shot pass (for rows that had no keyword match)
    if needs_zs and zero_shot_model:
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if torch.cuda.is_available() else -1
            logger.info(
                "Running zero-shot categorisation on %d rows (device=%s)…",
                len(needs_zs),
                "GPU" if device == 0 else "CPU",
            )
            zs = hf_pipeline("zero-shot-classification", model=zero_shot_model, device=device)

            batch_texts = [texts[i] for i in needs_zs]
            for start in range(0, len(batch_texts), zero_shot_batch_size):
                batch = batch_texts[start: start + zero_shot_batch_size]
                raw = zs(batch, zs_candidates, multi_label=True)
                if isinstance(raw, dict):
                    raw = [raw]
                for j, res in enumerate(raw):
                    gi = needs_zs[start + j]
                    threshold = cfg.get("zero_shot", {}).get("confidence_threshold", 0.30)
                    matched_cats = [
                        res["labels"][k]
                        for k, score in enumerate(res["scores"])
                        if score >= threshold
                    ]
                    results[gi] = "|".join(matched_cats) if matched_cats else "Statistical_Outlier_Only"
        except Exception as exc:
            logger.warning("Zero-shot categorisation failed (%s). Falling back.", exc)
            for i in needs_zs:
                results[i] = "Statistical_Outlier_Only"
    else:
        for i in needs_zs:
            results[i] = "Statistical_Outlier_Only"

    return results


# ---------------------------------------------------------------------------
# Category summary helpers
# ---------------------------------------------------------------------------

def category_breakdown(minority_df: pd.DataFrame) -> dict[str, int]:
    """Return a count dict of each category across flagged rows."""
    if "minority_category" not in minority_df.columns:
        return {}
    counts: dict[str, int] = {}
    for cats_str in minority_df["minority_category"].dropna():
        for cat in str(cats_str).split("|"):
            cat = cat.strip()
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def is_real_minority(row: pd.Series) -> bool:
    """True if flagged AND has a meaningful category (not just Statistical_Outlier_Only)."""
    if not row.get("is_minority_pattern", False):
        return False
    cats = str(row.get("minority_category", ""))
    return bool(cats) and "Statistical_Outlier_Only" not in cats
