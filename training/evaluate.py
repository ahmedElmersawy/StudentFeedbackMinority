"""
Feedback Atlas — Evaluation Script
====================================
Evaluate a trained model against a CSV dataset.

Usage:
  python -m training.evaluate --model-dir final_feedback_classifier --csv studentdataset.csv
  python -m training.evaluate --model-dir catme_feedback_classifier  --csv CATMEcomments_Training.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def evaluate(
    model_dir: str | Path,
    csv_path: str | Path,
    text_cols: list[str] | None = None,
    label_col: str | None = None,
    rating_col: str | None = None,
    batch_size: int = 32,
    output_json: str | Path | None = None,
) -> dict:
    """
    Load model, classify *csv_path*, compare against ground-truth labels.
    Prints a full classification report and optionally saves metrics to JSON.
    """
    from training.train import (
        auto_detect_label_column,
        auto_detect_rating_column,
        auto_detect_text_columns,
        derive_labels_from_ratings,
        load_csv,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ── Load data ─────────────────────────────────────────────────────────
    csv_path = Path(csv_path)
    model_dir = Path(model_dir)
    df = load_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    # ── Detect columns ────────────────────────────────────────────────────
    text_cols = text_cols or auto_detect_text_columns(df)
    df["text"] = (
        df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df[df["text"].str.len() > 5].reset_index(drop=True)

    has_labels = False
    true_labels: list[str] = []

    if label_col and label_col in df.columns:
        true_labels = df[label_col].astype(str).str.strip().tolist()
        has_labels = True
    elif rating_col and rating_col in df.columns:
        true_labels = derive_labels_from_ratings(df, rating_col).dropna().astype(str).tolist()
        has_labels = True
    else:
        # Try auto-detect
        det_lc = auto_detect_label_column(df, text_cols)
        det_rc = auto_detect_rating_column(df, text_cols)
        if det_lc:
            true_labels = df[det_lc].astype(str).str.strip().tolist()
            has_labels = True
        elif det_rc:
            true_labels = derive_labels_from_ratings(df, det_rc).dropna().astype(str).tolist()
            has_labels = True

    if not has_labels:
        logger.warning("No ground-truth labels found. Showing prediction distribution only.")

    # ── Load model ────────────────────────────────────────────────────────
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"No config.json in {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ── Predict ───────────────────────────────────────────────────────────
    texts = df["text"].tolist()
    predictions: list[str] = []
    confidences: list[float] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1).cpu().tolist()
            conf = torch.max(probs, dim=1).values.cpu().tolist()
            predictions.extend([id2label[p] for p in pred_ids])
            confidences.extend(conf)

    logger.info("Prediction distribution:")
    dist = pd.Series(predictions).value_counts()
    for lbl, cnt in dist.items():
        logger.info("  %-28s %5d  (%.1f%%)", lbl, cnt, 100.0 * cnt / len(predictions))

    metrics: dict = {
        "total": len(predictions),
        "avg_confidence": float(np.mean(confidences)),
        "label_distribution": dist.to_dict(),
    }

    if has_labels and len(true_labels) == len(predictions):
        unique_labels = sorted(set(true_labels + predictions))
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(true_labels, predictions, labels=unique_labels, digits=4))

        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        print("CONFUSION MATRIX")
        print("Labels:", unique_labels)
        print(cm)

        from sklearn.metrics import f1_score as _f1
        macro_f1 = float(_f1(true_labels, predictions, average="macro", zero_division=0))
        weighted_f1 = float(_f1(true_labels, predictions, average="weighted", zero_division=0))
        per_class = dict(zip(
            unique_labels,
            [float(v) for v in _f1(true_labels, predictions, labels=unique_labels, average=None, zero_division=0)],
        ))

        metrics.update({
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
            "f1_per_class": per_class,
        })

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", output_json)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained Feedback Atlas model.")
    parser.add_argument("--model-dir", default="final_feedback_classifier")
    parser.add_argument("--csv", dest="csv_path", required=True)
    parser.add_argument("--text-cols", nargs="*", default=None, metavar="COL")
    parser.add_argument("--label-col", default=None, metavar="COL")
    parser.add_argument("--rating-col", default=None, metavar="COL")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    args = parse_args()
    evaluate(
        model_dir=args.model_dir,
        csv_path=args.csv_path,
        text_cols=args.text_cols,
        label_col=args.label_col,
        rating_col=args.rating_col,
        batch_size=args.batch_size,
        output_json=args.output_json,
    )
