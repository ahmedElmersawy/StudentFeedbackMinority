"""
Feedback Atlas Training Script — Dual-Mode
==========================================
Trains distilroberta-base on student feedback.

Two modes:
  student_to_student  → CATME peer/self feedback (11 labels)
  student_to_professor → RMP + courseeval (24 labels)

Usage:
  # CATME Student→Student model
  python -m training.train --mode student_to_student

  # Professor model (RMP training + courseeval validation)
  python -m training.train --mode student_to_professor

  # Custom CSV (auto-detect labels from ratings or zero-shot)
  python -m training.train --csv mydata.csv --output-dir my_classifier

DO NOT CHANGE:
  - distilroberta-base base model
  - Weighted cross-entropy (BalancedTrainer)
  - Early stopping on minority_f1_mean
  - all-MiniLM-L6-v2 for embeddings (minority detection)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "backend" / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


# ---------------------------------------------------------------------------
# CATME subtype detection (Step 2 from spec)
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
# Legacy column detection helpers (kept for --csv mode)
# ---------------------------------------------------------------------------

_TEXT_HINTS = (
    "feedback", "comment", "review", "text", "opinion", "response",
    "answer", "open", "written", "notes", "peer", "team",
    "teaching", "course", "exam", "lab",
)


def _mean_len(s: pd.Series) -> float:
    s = s.dropna().astype(str)
    return float(s.str.len().mean()) if not s.empty else 0.0


def auto_detect_text_columns(df: pd.DataFrame, max_cols: int = 12) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for col in df.columns:
        dt = df[col].dtype
        if not pd.api.types.is_string_dtype(dt) and dt != object:
            continue
        ml = _mean_len(df[col])
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
        out = [c for c in df.columns
               if pd.api.types.is_string_dtype(df[c].dtype) or df[c].dtype == object][:max_cols]
    return out


def auto_detect_rating_column(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    for col in df.columns:
        if col in text_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            if vals.min() >= 1 and vals.max() <= 10:
                return col
            if set(vals.unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
                return col
    return None


def auto_detect_label_column(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    for col in df.columns:
        if col in text_cols:
            continue
        cl = col.lower()
        if cl in ("sentiment", "label", "class", "category"):
            return col
        if pd.api.types.is_string_dtype(df[col].dtype) or df[col].dtype == object:
            unique_vals = set(df[col].dropna().astype(str).str.lower().unique())
            if unique_vals and unique_vals.issubset({"positive", "negative", "neutral"}):
                return col
    return None


def _looks_headerless(val: str) -> bool:
    v = str(val).strip()
    return len(v) > 30 and " " in v


def load_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, nrows=2)
    if _looks_headerless(str(raw.iloc[0, 0])):
        logger.info("[load] Headerless CSV → adding 'text' header.")
        return pd.read_csv(path, header=None, names=["text"])
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Rating → sentiment
# ---------------------------------------------------------------------------

def rating_to_sentiment(rating: float, q33: float = 0.0, q66: float = 0.0, use_quantile: bool = False) -> str | None:
    if pd.isna(rating):
        return None
    if use_quantile:
        if rating <= q33:
            return "Negative"
        if rating <= q66:
            return "Neutral"
        return "Positive"
    if rating < 2.5:
        return "Negative"
    if rating < 3.8:
        return "Neutral"
    return "Positive"


def derive_labels_from_ratings(df: pd.DataFrame, rating_col: str) -> pd.Series:
    ratings = pd.to_numeric(df[rating_col], errors="coerce")
    vals = ratings.dropna()

    if len(vals) > 0 and set(vals.unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
        logger.info("[labels] Ternary (-1/0/1) scale → direct mapping.")
        def _ternary(r: float) -> str | None:
            if pd.isna(r):
                return None
            return "Negative" if r < 0 else ("Neutral" if r == 0 else "Positive")
        return ratings.apply(_ternary)

    sentiments = ratings.apply(rating_to_sentiment)

    if sentiments.nunique(dropna=True) <= 1:
        logger.info("[labels] Single-class → switching to quantile split.")
        q33 = float(ratings.quantile(0.33))
        q66 = float(ratings.quantile(0.66))
        sentiments = ratings.apply(rating_to_sentiment, q33=q33, q66=q66, use_quantile=True)

    return sentiments


# ---------------------------------------------------------------------------
# Zero-shot labeling
# ---------------------------------------------------------------------------

def zero_shot_label(
    texts: list[str],
    candidate_labels: list[str],
    label_map: dict[str, str] | None = None,
    cache_path: Path | None = None,
    batch_size: int = 32,
    zs_model: str = "facebook/bart-large-mnli",
) -> list[str]:
    """
    Assign labels via BART-MNLI zero-shot classification.
    Results are checkpointed to *cache_path* so long runs can resume.
    If label_map is provided, maps candidate strings → final label names.
    """
    import torch
    from transformers import pipeline as hf_pipeline

    cache_file = cache_path or (_PROJECT_ROOT / "zero_shot_labels_cache.json")
    cached: dict[str, str] = {}
    if cache_file.exists():
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        logger.info("[zero-shot] Loaded %d cached labels.", len(cached))

    device = 0 if torch.cuda.is_available() else -1
    logger.info("[zero-shot] Loading %s (device=%s)…", zs_model, "GPU" if device == 0 else "CPU")
    zs = hf_pipeline("zero-shot-classification", model=zs_model, device=device)

    results: list[str] = [""] * len(texts)
    to_label = [i for i, t in enumerate(texts) if t not in cached and t.strip()]

    for i, t in enumerate(texts):
        if t in cached:
            results[i] = cached[t]

    total = len(to_label)
    for start in range(0, total, batch_size):
        batch_idx = to_label[start: start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        raw = zs(batch_texts, candidate_labels, multi_label=False)
        if isinstance(raw, dict):
            raw = [raw]
        for j, res in enumerate(raw):
            gi = batch_idx[j]
            winner = res["labels"][0]
            final = label_map.get(winner, winner) if label_map else winner
            results[gi] = final
            cached[texts[gi]] = final

        done = start + batch_size
        if done % 1000 == 0 or done >= total:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            logger.info("[zero-shot] %d/%d (checkpoint saved).", min(done, total), total)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cached, f)

    return results


def detect_dataset_type(texts: list[str]) -> str:
    sample = " ".join(texts[:300]).lower()
    peer_signals = ["teammate", "team member", "group member", "contributed", "catme", "worked together"]
    course_signals = ["professor", "lecture", "syllabus", "course material", "exam", "assignment"]
    ps = sum(s in sample for s in peer_signals)
    cs = sum(s in sample for s in course_signals)
    return "peer" if ps >= cs else "course"


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class FeedbackDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx].clone().detach() for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ---------------------------------------------------------------------------
# Weighted loss trainer (DO NOT CHANGE — spec constraint)
# ---------------------------------------------------------------------------

class BalancedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, num_labels: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=self._class_weights.to(model.device))(
            outputs.logits.view(-1, self._num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Student→Student (CATME) training
# ---------------------------------------------------------------------------

def _prepare_catme_data(cfg: dict) -> pd.DataFrame:
    """Load CATME, detect subtypes, assign zero-shot labels per subtype."""
    from .data_loaders import load_catme

    df = load_catme()

    # Detect subtype
    logger.info("[catme] Detecting self_assessment vs peer_feedback subtypes…")
    df["subtype"] = df["text"].apply(detect_catme_subtype)
    counts = df["subtype"].value_counts()
    logger.info("[catme] Subtypes: %s", counts.to_dict())

    # Anonymize if possible
    try:
        from backend.anonymizer import anonymize_series
        placeholder = cfg.get("anonymization", {}).get("placeholder", "[STUDENT]")
        spacy_model = cfg.get("anonymization", {}).get("spacy_model", "en_core_web_sm")
        df["text"] = anonymize_series(df["text"], placeholder=placeholder, spacy_model=spacy_model)
        logger.info("[catme] Anonymization applied.")
    except Exception as exc:
        logger.warning("[catme] Anonymization skipped: %s", exc)

    cfg_zs = cfg.get("zero_shot", {})
    zs_model = cfg_zs.get("model", "facebook/bart-large-mnli")
    bs = cfg_zs.get("batch_size", 32)
    zs_candidates = cfg.get("zero_shot_candidates", {})
    zs_label_map = cfg.get("zero_shot_label_map", {})

    # Zero-shot label each subtype separately
    labels_out = [""] * len(df)

    for subtype in ("peer_feedback", "self_assessment"):
        mask = df["subtype"] == subtype
        idxs = df.index[mask].tolist()
        texts = df.loc[mask, "text"].tolist()
        if not texts:
            continue
        candidates = zs_candidates.get(subtype, ["Positive", "Neutral", "Negative"])
        lmap = zs_label_map.get(subtype, {})
        cache_path = _PROJECT_ROOT / f"zero_shot_cache_{subtype}.json"
        logger.info("[catme] Zero-shot labeling %d %s rows…", len(texts), subtype)
        batch_labels = zero_shot_label(
            texts,
            candidate_labels=candidates,
            label_map=lmap,
            cache_path=cache_path,
            batch_size=bs,
            zs_model=zs_model,
        )
        for i, idx in enumerate(idxs):
            labels_out[df.index.get_loc(idx)] = batch_labels[i]

    df["sentiment"] = labels_out

    # Sanity check: no single label should dominate by more than 30%.
    # If it does, the zero-shot candidates are likely miscalibrated.
    dist = df["sentiment"].value_counts(normalize=True)
    for label, pct in dist.items():
        if pct > 0.30:
            logger.warning(
                "[catme] LABEL IMBALANCE WARNING: '%s' = %.1f%% of training data — "
                "check zero-shot candidates. Expected ≤30%%.",
                label, pct * 100,
            )

    logger.info("[catme] Label distribution after zero-shot labeling:")
    for label, pct in dist.items():
        logger.info("  %-40s %.1f%%", label, pct * 100)

    return df


def train_catme(output_dir: str | Path | None = None) -> Path:
    """Train Student→Student classifier on CATME data."""
    cfg = load_config()
    out_dir = Path(output_dir or (_PROJECT_ROOT / cfg.get("model", {}).get("catme_output_dir", "catme_feedback_classifier")))
    df = _prepare_catme_data(cfg)
    return _run_training(df, out_dir, cfg, source_name="CATME")


# ---------------------------------------------------------------------------
# Student→Professor training
# ---------------------------------------------------------------------------

def _prepare_professor_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load RMP (training) + courseeval (validation).
    RMP rows get pseudo-labels from emotional_label, then refined with zero-shot.
    courseeval is returned as a held-out validation set only.
    """
    from .data_loaders import load_professor_combined

    train_df, val_df = load_professor_combined(cfg=cfg, include_courseeval_in_train=True)

    cfg_zs = cfg.get("zero_shot", {})
    zs_model = cfg_zs.get("model", "facebook/bart-large-mnli")
    bs = cfg_zs.get("batch_size", 32)
    candidates = cfg.get("zero_shot_candidates", {}).get("professor", [])
    lmap = cfg.get("zero_shot_label_map", {}).get("professor", {})
    rmp_lmap = cfg.get("rmp_label_map", {})

    # RMP: run zero-shot labeling over all rows using the full 24-label candidate set.
    # Do NOT pre-seed from emotional_label — it only covers 5 rough labels and prevents
    # BART-MNLI from assigning the other 19 professor categories.
    # The cache checkpoint mechanism handles restarts cleanly.
    logger.info("[professor] Zero-shot labeling %d RMP rows (24-class)…", len(train_df))
    cache_path = _PROJECT_ROOT / "zero_shot_cache_professor.json"

    texts = train_df["text"].fillna("").astype(str).tolist()
    labels = zero_shot_label(
        texts,
        candidate_labels=candidates,
        label_map=lmap,
        cache_path=cache_path,
        batch_size=bs,
        zs_model=zs_model,
    )
    train_df = train_df.copy()
    train_df["sentiment"] = labels

    # courseeval validation: use dimension-constrained zero-shot so val labels
    # are consistent with training labels (not just broad Positive/Neutral/Negative)
    if "sentiment" not in val_df.columns:
        from .data_loaders import load_courseeval_labeled
        try:
            val_labeled = load_courseeval_labeled(cfg=cfg, zs_model=cfg_zs.get("model", "facebook/bart-large-mnli"))
            val_df = val_df.copy()
            val_df["sentiment"] = val_labeled.loc[val_df.index, "sentiment"].values
        except Exception as exc:
            logger.warning("[professor] Val labeling failed (%s) — falling back to rating buckets.", exc)
            val_df = val_df.copy()
            ratings = pd.to_numeric(val_df.get("rating", pd.Series(dtype=float)), errors="coerce")
            val_df["sentiment"] = ratings.apply(
                lambda r: None if pd.isna(r) else ("Negative" if r < 0 else ("Neutral" if r == 0 else "Positive"))
            )

    return train_df, val_df


def train_professor(output_dir: str | Path | None = None) -> Path:
    """Train Student→Professor classifier on RMP, validate on courseeval."""
    cfg = load_config()
    out_dir = Path(output_dir or (_PROJECT_ROOT / cfg.get("model", {}).get("professor_output_dir", "professor_feedback_classifier")))

    train_df, val_df = _prepare_professor_data(cfg)

    # Train on RMP
    saved = _run_training(train_df, out_dir, cfg, source_name="RMP+courseeval", val_df=val_df)

    # Report courseeval validation accuracy
    _validate_on_courseeval(saved, val_df)
    return saved


def _validate_on_courseeval(model_dir: Path, val_df: pd.DataFrame) -> None:
    """Run inference on courseeval and log per-label precision/recall."""
    try:
        from backend.pipeline import run_inference
        df = val_df[val_df["text"].notna() & (val_df["text"].str.len() > 5)].copy()
        result = run_inference(df, model_dir=str(model_dir))
        if "sentiment" in result.columns and "prediction" in result.columns:
            mask = result["sentiment"].notna()
            logger.info(
                "\n[courseeval validation]\n%s",
                classification_report(
                    result.loc[mask, "sentiment"],
                    result.loc[mask, "prediction"],
                    zero_division=0,
                ),
            )
    except Exception as exc:
        logger.warning("[courseeval validation] Skipped: %s", exc)


# ---------------------------------------------------------------------------
# Core training function (shared by both modes)
# ---------------------------------------------------------------------------

def _run_training(
    df: pd.DataFrame,
    output_dir: Path,
    cfg: dict,
    source_name: str = "dataset",
    val_df: pd.DataFrame | None = None,
) -> Path:
    """
    Given a DataFrame with 'text' + 'sentiment' columns, train and save a model.
    If val_df is provided it is used as the held-out eval set; otherwise a
    random split of df is used.
    """
    tc = cfg.get("training", {})
    mc = cfg.get("model", {})
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean
    df = df[df["text"].notna() & (df["text"].str.len() > 10)].copy()
    df = df[df["sentiment"].notna()].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df = df[df["sentiment"] != "nan"].copy()

    label_counts = df["sentiment"].value_counts()
    logger.info("=" * 70)
    logger.info("LABEL DISTRIBUTION (%s — before undersampling)", source_name)
    logger.info("=" * 70)
    for lbl, cnt in label_counts.items():
        logger.info("  %-36s %5d  (%5.1f%%)", lbl, cnt, 100.0 * cnt / len(df))

    # Undersample every class that exceeds (second_largest × undersample_ratio).
    # Applying the cap to ALL over-represented labels prevents a mislabeled class
    # (e.g. Minority_Peer_Experience inflated by a bad zero-shot cache) from
    # skewing the training distribution even when it is not the single dominant class.
    undersample_ratio = tc.get("undersample_ratio", None)
    if undersample_ratio is not None and len(label_counts) > 1:
        sorted_counts = label_counts.sort_values(ascending=False)
        second = int(sorted_counts.iloc[1])
        cap = int(second * undersample_ratio)
        rng = tc.get("random_state", 42)
        parts = []
        for lbl, cnt in label_counts.items():
            subset = df[df["sentiment"] == lbl]
            if cnt > cap:
                subset = subset.sample(cap, random_state=rng)
                logger.info("Undersampled '%s' %d → %d.", lbl, cnt, cap)
            parts.append(subset)
        df = pd.concat(parts).sample(frac=1, random_state=rng).reset_index(drop=True)
        label_counts = df["sentiment"].value_counts()

    if len(label_counts) < 2:
        raise ValueError(f"Need ≥2 classes; got {len(label_counts)}.")

    # Encode labels
    unique_labels = sorted(df["sentiment"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = df["sentiment"].map(label2id)

    X = df["text"].tolist()
    y = df["label_id"].tolist()

    # Split
    X_train, X_val_split, y_train, y_val_split = train_test_split(
        X, y,
        test_size=tc.get("test_size", 0.2),
        stratify=y,
        random_state=tc.get("random_state", 42),
    )

    # Use external val_df if provided (e.g. courseeval), otherwise use split
    if val_df is not None and "text" in val_df.columns and "sentiment" in val_df.columns:
        vdf = val_df[val_df["text"].notna() & (val_df["sentiment"].notna())].copy()
        vdf["text"] = vdf["text"].astype(str).str.strip()
        vdf["sentiment"] = vdf["sentiment"].astype(str).str.strip()
        # Only keep labels present in training set
        vdf = vdf[vdf["sentiment"].isin(label2id)]
        if len(vdf) > 0:
            X_val_split = vdf["text"].tolist()
            y_val_split = vdf["sentiment"].map(label2id).tolist()
            logger.info("[val] Using external validation set: %d rows.", len(X_val_split))

    logger.info("Train: %d  |  Val: %d", len(X_train), len(X_val_split))

    # Tokenizer
    base_model = mc.get("base_model", "distilroberta-base")
    max_length = mc.get("max_length", 256)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_ds = FeedbackDataset(X_train, y_train, tokenizer, max_length=max_length)
    val_ds = FeedbackDataset(X_val_split, y_val_split, tokenizer, max_length=max_length)

    # Class weights — boost minority classes
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)

    # Extra multiplier for true minority classes.
    # Labels in non_minority_labels (e.g. Majority_Positive, Self_Positive) are
    # intentionally common — do NOT boost them or the model over-predicts them.
    mult = tc.get("minority_class_weight_multiplier", 3.0)
    non_minority_label_names = set(tc.get("non_minority_labels", []))
    max_cnt = max(Counter(y_train).values())
    minority_ids = {
        i for i, cnt in Counter(y_train).items()
        if cnt < max_cnt and id2label[i] not in non_minority_label_names
    }
    weights = [
        w * mult if i in minority_ids else w
        for i, w in enumerate(weights)
    ]
    class_weights = torch.tensor(weights, dtype=torch.float)
    logger.info("Class weights: %s", {id2label[c]: round(weights[j], 3) for j, c in enumerate(classes)})

    # Metrics — early stopping on minority_f1_mean (spec constraint)
    minority_class_ids = sorted(minority_ids)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        unique_eval = np.unique(labels)
        report = classification_report(labels, preds, labels=unique_eval, output_dict=True, zero_division=0)
        macro = report["macro avg"]["f1-score"]
        weighted = report["weighted avg"]["f1-score"]
        min_f1_scores = [
            report[str(i)]["f1-score"]
            for i in minority_class_ids
            if str(i) in report and i in unique_eval
        ]
        minority_f1 = float(np.mean(min_f1_scores)) if min_f1_scores else 0.0
        return {"f1_macro": macro, "f1_weighted": weighted, "minority_f1_mean": minority_f1}

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=tc.get("num_epochs", 15),
        per_device_train_batch_size=tc.get("batch_size", 16),
        per_device_eval_batch_size=tc.get("batch_size", 16),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=tc.get("early_stopping_metric", "minority_f1_mean"),
        greater_is_better=True,
        learning_rate=tc.get("learning_rate", 3e-5),
        warmup_ratio=tc.get("warmup_ratio", 0.1),
        weight_decay=tc.get("weight_decay", 0.01),
        fp16=tc.get("fp16", torch.cuda.is_available()),
        report_to=[],
        save_total_limit=2,
        seed=tc.get("random_state", 42),
    )

    # Differential learning rates (spec: body_lr=1e-5, head_lr=1e-3)
    no_decay = ["bias", "LayerNorm.weight"]
    body_lr = tc.get("body_lr", 1e-5)
    head_lr = tc.get("head_lr", 1e-3)
    optimizer = AdamW([
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and "classifier" not in n],
            "lr": body_lr, "weight_decay": tc.get("weight_decay", 0.01),
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and "classifier" not in n],
            "lr": body_lr, "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": head_lr, "weight_decay": tc.get("weight_decay", 0.01),
        },
    ])

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=tc.get("early_stopping_patience", 3)
        )],
        optimizers=(optimizer, None),
        class_weights=class_weights,
        num_labels=len(unique_labels),
    )

    logger.info("=" * 70)
    logger.info("TRAINING STARTED — %s", source_name)
    logger.info("=" * 70)
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    logger.info("Final validation results:")
    for k, v in results.items():
        if "runtime" not in k and "per_second" not in k:
            logger.info("  %-36s %.4f", k, v)

    preds_raw = trainer.predict(val_ds).predictions
    pred_labels = np.argmax(preds_raw, axis=1)
    logger.info("\n%s", classification_report(y_val_split, pred_labels, target_names=unique_labels, digits=4))

    # Save
    logger.info("=" * 70)
    logger.info("SAVING MODEL → %s", output_dir)
    logger.info("=" * 70)

    try:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    except Exception as exc:
        fallback = output_dir.parent / (output_dir.name + "_fallback")
        logger.warning("Primary save failed (%s) → %s", exc, fallback)
        trainer.save_model(str(fallback))
        tokenizer.save_pretrained(str(fallback))
        output_dir = fallback

    with open(output_dir / "label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "source": source_name,
            "num_labels": len(unique_labels),
            "labels": unique_labels,
            "eval_f1_macro": results.get("eval_f1_macro", 0),
            "eval_minority_f1_mean": results.get("eval_minority_f1_mean", 0),
        }, f, indent=2)

    logger.info("Saved: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Legacy CSV training (backward compat — for --csv flag)
# ---------------------------------------------------------------------------

def train(
    csv_path: str | Path,
    output_dir: str | Path,
    text_cols: list[str] | None = None,
    rating_col: str | None = None,
    label_col: str | None = None,
    anonymize: bool = True,
    use_zero_shot_for_labels: bool = True,
    zero_shot_cache: str | Path | None = None,
) -> Path:
    """Legacy single-CSV training (original Purdue dataset, any CSV)."""
    cfg = load_config()
    tc = cfg.get("training", {})
    mc = cfg.get("model", {})

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    df_raw = load_csv(csv_path)
    logger.info("Rows: %d, Columns: %s", len(df_raw), df_raw.columns.tolist())

    detected_text_cols = text_cols or auto_detect_text_columns(df_raw)
    detected_rating_col = rating_col or (None if label_col else auto_detect_rating_column(df_raw, detected_text_cols))
    detected_label_col = label_col or auto_detect_label_column(df_raw, detected_text_cols)

    df = df_raw.copy()
    df["text"] = (
        df[detected_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

    if anonymize:
        try:
            sys.path.insert(0, str(_PROJECT_ROOT))
            from backend.anonymizer import anonymize_series
            placeholder = cfg.get("anonymization", {}).get("placeholder", "[STUDENT]")
            spacy_model = cfg.get("anonymization", {}).get("spacy_model", "en_core_web_sm")
            df["text"] = anonymize_series(df["text"], placeholder=placeholder, spacy_model=spacy_model)
        except Exception as exc:
            logger.warning("Anonymization skipped: %s", exc)

    if detected_label_col:
        df["sentiment"] = df_raw[detected_label_col].astype(str).str.strip()
    elif detected_rating_col:
        df["sentiment"] = derive_labels_from_ratings(df, detected_rating_col)
    elif use_zero_shot_for_labels:
        dataset_type = detect_dataset_type(df["text"].dropna().tolist())
        cfg_zs = cfg.get("zero_shot", {})
        cache = Path(zero_shot_cache or cfg_zs.get("cache_path", _PROJECT_ROOT / "zero_shot_labels_cache.json"))
        cfg_labels = cfg.get("labels", {})
        candidates = cfg_labels.get(dataset_type, cfg_labels.get("broad", ["Positive", "Neutral", "Negative"]))
        df["sentiment"] = zero_shot_label(
            df["text"].fillna("").astype(str).tolist(),
            candidate_labels=candidates,
            cache_path=cache,
            batch_size=cfg_zs.get("batch_size", 32),
            zs_model=cfg_zs.get("model", "facebook/bart-large-mnli"),
        )
    else:
        raise ValueError("No labels available.")

    return _run_training(df, output_dir, cfg, source_name=csv_path.stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Feedback Atlas — train sentiment classifier.")
    parser.add_argument(
        "--mode",
        choices=["student_to_student", "student_to_professor"],
        default=None,
        help="Dual-mode training: student_to_student (CATME) or student_to_professor (RMP+courseeval).",
    )
    parser.add_argument("--csv", dest="csv_path", default=None,
                        help="Legacy: train on a single CSV file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--text-cols", nargs="*", default=None, metavar="COL")
    parser.add_argument("--rating-col", default=None, metavar="COL")
    parser.add_argument("--label-col", default=None, metavar="COL")
    parser.add_argument("--no-anonymize", action="store_true")
    parser.add_argument("--no-zero-shot", action="store_true")
    parser.add_argument("--zero-shot-cache", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config()

    if args.mode == "student_to_student":
        out = args.output_dir or str(_PROJECT_ROOT / cfg.get("model", {}).get("catme_output_dir", "catme_feedback_classifier"))
        train_catme(output_dir=out)

    elif args.mode == "student_to_professor":
        out = args.output_dir or str(_PROJECT_ROOT / cfg.get("model", {}).get("professor_output_dir", "professor_feedback_classifier"))
        train_professor(output_dir=out)

    elif args.csv_path:
        csv_stem = Path(args.csv_path).stem.lower()
        if args.output_dir:
            out_dir = args.output_dir
        elif "catme" in csv_stem:
            out_dir = str(_PROJECT_ROOT / cfg.get("model", {}).get("catme_output_dir", "catme_feedback_classifier"))
        else:
            out_dir = str(_PROJECT_ROOT / "custom_classifier")

        train(
            csv_path=args.csv_path,
            output_dir=out_dir,
            text_cols=args.text_cols,
            rating_col=args.rating_col,
            label_col=args.label_col,
            anonymize=not args.no_anonymize,
            use_zero_shot_for_labels=not args.no_zero_shot,
            zero_shot_cache=args.zero_shot_cache,
        )
    else:
        parser = argparse.ArgumentParser()
        parser.error("Provide --mode (student_to_student | student_to_professor) or --csv <path>.")
