"""
Feedback Atlas Training Script
================================
Trains a distilroberta-base sentiment classifier on any feedback CSV.
Supports:
  - studentdataset.csv  (rating-based label derivation)
  - CATMEcomments_Training.csv  (headerless; zero-shot label generation)
  - Any CSV with a label column, rating column, or pure text

Usage examples:
  # Original Purdue dataset (auto-detect .1 text cols + numeric ratings)
  python -m training.train --csv studentdataset.csv

  # CATME peer-feedback (headerless, zero-shot labels, anonymize names)
  python -m training.train --csv CATMEcomments_Training.csv \
      --output-dir catme_feedback_classifier

  # CSV with explicit label column
  python -m training.train --csv mydata.csv --label-col sentiment

  # Explicit text + rating columns
  python -m training.train --csv mydata.csv \
      --text-cols "Q1 Comments" "Q2 Comments" --rating-col avg_score
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
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
# Column detection (mirrors backend/pipeline.py but standalone)
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
        out = [c for c in df.columns if pd.api.types.is_string_dtype(df[c].dtype) or df[c].dtype == object][:max_cols]
    return out


def auto_detect_rating_column(df: pd.DataFrame, text_cols: list[str]) -> str | None:
    for col in df.columns:
        if col in text_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            # Standard 1-10 scale
            if vals.min() >= 1 and vals.max() <= 10:
                return col
            # Ternary -1/0/1 scale
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


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _looks_headerless(val: str) -> bool:
    v = str(val).strip()
    return len(v) > 30 and " " in v


def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV; auto-handle headerless CATME-style files."""
    raw = pd.read_csv(path, header=None, nrows=2)
    if _looks_headerless(str(raw.iloc[0, 0])):
        logger.info("[load] Headerless CSV detected → adding 'text' header.")
        return pd.read_csv(path, header=None, names=["text"])
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Label derivation
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

    # Ternary -1/0/1 scale: map directly without threshold guessing.
    if len(vals) > 0 and set(vals.unique()).issubset({-1, 0, 1, -1.0, 0.0, 1.0}):
        logger.info("[labels] Detected ternary (-1/0/1) rating scale → direct mapping.")
        def _ternary(r: float) -> str | None:
            if pd.isna(r):
                return None
            return "Negative" if r < 0 else ("Neutral" if r == 0 else "Positive")
        return ratings.apply(_ternary)

    sentiments = ratings.apply(rating_to_sentiment)

    if sentiments.nunique(dropna=True) <= 1:
        logger.info("[labels] Single-class from thresholds → switching to quantile split.")
        q33 = float(ratings.quantile(0.33))
        q66 = float(ratings.quantile(0.66))
        sentiments = ratings.apply(rating_to_sentiment, q33=q33, q66=q66, use_quantile=True)

    return sentiments


def zero_shot_label(
    texts: list[str],
    dataset_type: str,
    cache_path: Path,
    batch_size: int = 32,
    zs_model: str = "facebook/bart-large-mnli",
    candidate_labels: list[str] | None = None,
) -> list[str]:
    """
    Assign labels to *texts* via BART-MNLI zero-shot classification.
    Results are checkpointed to *cache_path* so long runs can resume.
    """
    import torch
    from transformers import pipeline as hf_pipeline

    # Load cache
    cached: dict[str, str] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        logger.info("[zero-shot] Loaded %d cached labels.", len(cached))

    if candidate_labels is None:
        cfg_labels = load_config().get("labels", {})
        candidate_labels = cfg_labels.get(dataset_type, cfg_labels.get("broad", ["Positive", "Neutral", "Negative"]))
    logger.info("[zero-shot] Candidate labels: %s", candidate_labels)

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
        batch_idx = to_label[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        raw = zs(batch_texts, candidate_labels, multi_label=False)
        if isinstance(raw, dict):
            raw = [raw]
        for j, res in enumerate(raw):
            gi = batch_idx[j]
            winner = res["labels"][0]
            results[gi] = winner
            cached[texts[gi]] = winner

        done = start + batch_size
        if done % 1000 == 0 or done >= total:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached, f)
            logger.info("[zero-shot] Progress: %d/%d (checkpoint saved).", min(done, total), total)

    with open(cache_path, "w", encoding="utf-8") as f:
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
# Weighted loss trainer
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
# Main training function
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
    """
    Full training pipeline.

    Returns the path to the saved model directory.
    """
    cfg = load_config()
    tc = cfg.get("training", {})
    mc = cfg.get("model", {})

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("LOADING DATA: %s", csv_path)
    logger.info("=" * 70)
    df_raw = load_csv(csv_path)
    logger.info("Rows: %d, Columns: %s", len(df_raw), df_raw.columns.tolist())

    # ── Detect columns ────────────────────────────────────────────────────
    detected_text_cols = text_cols or auto_detect_text_columns(df_raw)
    if not detected_text_cols:
        raise ValueError("No text columns found. Pass --text-cols explicitly.")
    detected_rating_col = rating_col or (None if label_col else auto_detect_rating_column(df_raw, detected_text_cols))
    detected_label_col = label_col or auto_detect_label_column(df_raw, detected_text_cols)

    logger.info("Text columns : %s", detected_text_cols)
    logger.info("Rating column: %s", detected_rating_col)
    logger.info("Label column : %s", detected_label_col)

    # ── Combine text ──────────────────────────────────────────────────────
    df = df_raw.copy()
    df["text"] = (
        df[detected_text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )

    # ── Anonymize ─────────────────────────────────────────────────────────
    if anonymize:
        try:
            sys.path.insert(0, str(_PROJECT_ROOT))
            from backend.anonymizer import anonymize_series
            placeholder = cfg.get("anonymization", {}).get("placeholder", "[STUDENT]")
            spacy_model = cfg.get("anonymization", {}).get("spacy_model", "en_core_web_sm")
            df["text"] = anonymize_series(df["text"], placeholder=placeholder, spacy_model=spacy_model)
            logger.info("Anonymization applied.")
        except Exception as exc:
            logger.warning("Anonymization skipped: %s", exc)

    # ── Labels ────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("DERIVING LABELS")
    logger.info("=" * 70)

    if detected_label_col:
        logger.info("Strategy: explicit label column '%s'", detected_label_col)
        df["sentiment"] = df_raw[detected_label_col].astype(str).str.strip()
    elif detected_rating_col:
        logger.info("Strategy: derive from rating column '%s'", detected_rating_col)
        df["sentiment"] = derive_labels_from_ratings(df, detected_rating_col)
    elif use_zero_shot_for_labels:
        dataset_type = detect_dataset_type(df["text"].dropna().tolist())
        logger.info("Strategy: zero-shot classification (dataset_type=%s)", dataset_type)
        cfg_zs = cfg.get("zero_shot", {})
        cache = Path(zero_shot_cache or cfg_zs.get("cache_path", _PROJECT_ROOT / "zero_shot_labels_cache.json"))
        labels = zero_shot_label(
            df["text"].fillna("").astype(str).tolist(),
            dataset_type=dataset_type,
            cache_path=cache,
            batch_size=cfg_zs.get("batch_size", 32),
            zs_model=cfg_zs.get("model", "facebook/bart-large-mnli"),
        )
        df["sentiment"] = labels
    else:
        raise ValueError(
            "No labels available. Pass --label-col, provide a rating column, "
            "or enable zero-shot with --use-zero-shot."
        )

    # ── Clean ─────────────────────────────────────────────────────────────
    df = df[df["text"].str.len() > 10].copy()
    df = df.dropna(subset=["sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip()

    label_counts = df["sentiment"].value_counts()
    logger.info("=" * 70)
    logger.info("LABEL DISTRIBUTION (before undersampling)")
    logger.info("=" * 70)
    for lbl, cnt in label_counts.items():
        logger.info("  %-28s %5d  (%5.1f%%)", lbl, cnt, 100.0 * cnt / len(df))

    # ── Undersample dominant class ─────────────────────────────────────────
    undersample_ratio = tc.get("undersample_ratio", None)
    if undersample_ratio is not None and len(label_counts) > 1:
        dominant = label_counts.idxmax()
        second_largest = int(label_counts.drop(dominant).max())
        cap = int(second_largest * undersample_ratio)
        if label_counts[dominant] > cap:
            keep_dominant = df[df["sentiment"] == dominant].sample(
                cap, random_state=tc.get("random_state", 42)
            )
            df = pd.concat(
                [keep_dominant, df[df["sentiment"] != dominant]]
            ).sample(frac=1, random_state=tc.get("random_state", 42)).reset_index(drop=True)
            label_counts = df["sentiment"].value_counts()
            logger.info(
                "Undersampled '%s': capped at %d (%.1f× second-largest class).",
                dominant, cap, undersample_ratio,
            )
            logger.info("=" * 70)
            logger.info("LABEL DISTRIBUTION (after undersampling)")
            logger.info("=" * 70)
            for lbl, cnt in label_counts.items():
                logger.info("  %-28s %5d  (%5.1f%%)", lbl, cnt, 100.0 * cnt / len(df))

    if len(label_counts) < 2:
        raise ValueError(f"Need at least 2 classes; got {len(label_counts)}.")

    # ── Encode labels ─────────────────────────────────────────────────────
    unique_labels = sorted(df["sentiment"].unique())
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = df["sentiment"].map(label2id)

    X = df["text"].tolist()
    y = df["label_id"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=tc.get("test_size", 0.2),
        stratify=y,
        random_state=tc.get("random_state", 42),
    )
    logger.info("Train: %d  |  Val: %d", len(X_train), len(X_val))

    # ── Tokenizer & datasets ──────────────────────────────────────────────
    base_model = mc.get("base_model", "distilroberta-base")
    max_length = mc.get("max_length", 256)
    logger.info("Loading tokenizer: %s", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    train_ds = FeedbackDataset(X_train, y_train, tokenizer, max_length=max_length)
    val_ds = FeedbackDataset(X_val, y_val, tokenizer, max_length=max_length)

    # ── Class weights ─────────────────────────────────────────────────────
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float)
    logger.info("Class weights: %s", dict(zip([id2label[c] for c in classes], weights.round(3))))

    minority_class_ids = [
        i for i, cnt in Counter(y_train).items() if cnt < max(Counter(y_train).values())
    ]

    # ── Metrics ───────────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Loading model: %s  (%d labels)", base_model, len(unique_labels))
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(unique_labels), id2label=id2label, label2id=label2id
    )

    # ── Training args ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=tc.get("num_epochs", 15),
        per_device_train_batch_size=tc.get("batch_size", 16),
        per_device_eval_batch_size=tc.get("batch_size", 16),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="minority_f1_mean",
        greater_is_better=True,
        learning_rate=tc.get("learning_rate", 3e-5),
        warmup_ratio=tc.get("warmup_ratio", 0.1),
        weight_decay=tc.get("weight_decay", 0.01),
        fp16=tc.get("fp16", torch.cuda.is_available()),
        report_to=[],
        save_total_limit=2,
        seed=tc.get("random_state", 42),
    )

    # Differential learning rates
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and "classifier" not in n],
         "lr": 1e-5, "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and "classifier" not in n],
         "lr": 1e-5, "weight_decay": 0.0},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n],
         "lr": 1e-3, "weight_decay": 0.01},
    ])

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tc.get("early_stopping_patience", 4))],
        optimizers=(optimizer, None),
        class_weights=class_weights,
        num_labels=len(unique_labels),
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("TRAINING STARTED")
    logger.info("=" * 70)
    trainer.train()

    # ── Evaluate ─────────────────────────────────────────────────────────
    results = trainer.evaluate()
    logger.info("Final validation results:")
    for k, v in results.items():
        if "runtime" not in k and "per_second" not in k:
            logger.info("  %-32s %.4f", k, v)

    preds_raw = trainer.predict(val_ds).predictions
    pred_labels = np.argmax(preds_raw, axis=1)
    logger.info("\n%s", classification_report(y_val, pred_labels, target_names=unique_labels, digits=4))

    # ── Save ─────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SAVING MODEL → %s", output_dir)
    logger.info("=" * 70)

    try:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
    except Exception as exc:
        fallback = output_dir.parent / (output_dir.name + "_fallback")
        logger.warning("Primary save failed (%s). Retrying → %s", exc, fallback)
        trainer.save_model(str(fallback))
        tokenizer.save_pretrained(str(fallback))
        output_dir = fallback

    with open(output_dir / "label_mappings.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "source_csv": str(csv_path),
            "text_cols": detected_text_cols,
            "label_strategy": "explicit" if detected_label_col else ("rating" if detected_rating_col else "zero_shot"),
            "num_labels": len(unique_labels),
            "labels": unique_labels,
            "eval_f1_macro": results.get("eval_f1_macro", 0),
            "eval_minority_f1_mean": results.get("eval_minority_f1_mean", 0),
        }, f, indent=2)

    logger.info("Saved: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Feedback Atlas — train sentiment classifier.")
    parser.add_argument("--csv", dest="csv_path", default="studentdataset.csv")
    parser.add_argument("--output-dir", default=None,
                        help="Model output directory (default: auto-named from CSV).")
    parser.add_argument("--text-cols", nargs="*", default=None, metavar="COL")
    parser.add_argument("--rating-col", default=None, metavar="COL")
    parser.add_argument("--label-col", default=None, metavar="COL")
    parser.add_argument("--no-anonymize", action="store_true")
    parser.add_argument("--no-zero-shot", action="store_true",
                        help="Disable zero-shot label generation (fail if no labels/ratings).")
    parser.add_argument("--zero-shot-cache", default=None,
                        help="Path to zero-shot label cache JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    csv_stem = Path(args.csv_path).stem.lower()
    if args.output_dir:
        out_dir = args.output_dir
    elif "catme" in csv_stem:
        out_dir = str(_PROJECT_ROOT / "catme_feedback_classifier")
    else:
        out_dir = str(_PROJECT_ROOT / "final_feedback_classifier")

    logger.info("Output directory: %s", out_dir)

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
