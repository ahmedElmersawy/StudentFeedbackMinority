"""
Quick evaluation of the professor feedback classifier on studentdataset.csv.
Tests on the 'teaching' text column (the dimension the model was trained for).
"""
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "professor_feedback_classifier"
CSV_PATH  = "studentdataset.csv"

# ----- load model -----------------------------------------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with open(f"{MODEL_DIR}/label_mappings.json") as f:
    mappings = json.load(f)
id2label = mappings["id2label"]

# ----- load data ------------------------------------------------------------
# CSV has two header rows (name, first-sample), so use header=[0] and skip oddly named cols
raw = pd.read_csv(CSV_PATH, header=[0, 1])
# Flatten multi-index: grab teaching text (column 1) and teaching label (column 0)
teach_text  = raw.iloc[:, 1].astype(str).str.strip()
teach_label = pd.to_numeric(raw.iloc[:, 0], errors="coerce")

# Map -1/0/1 → Negative/Neutral/Positive for reference
def ternary(r):
    if pd.isna(r): return None
    return "Negative" if r < 0 else ("Neutral" if r == 0 else "Positive")

ref_labels = teach_label.apply(ternary)

# Drop rows with missing text or label
mask = teach_text.notna() & (teach_text != "") & ref_labels.notna()
texts  = teach_text[mask].tolist()
refs   = ref_labels[mask].tolist()
print(f"\nRows with valid teaching text: {len(texts)}")

# ----- inference ------------------------------------------------------------
def predict_batch(texts, batch_size=32):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, truncation=True, padding=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        ids = logits.argmax(dim=-1).cpu().tolist()
        preds.extend([id2label[str(i)] for i in ids])
    return preds

print("Running inference...")
predictions = predict_batch(texts)

# ----- results --------------------------------------------------------------
print("\n" + "="*60)
print("MODEL PREDICTIONS  (professor labels)")
print("="*60)
from collections import Counter
print("Predicted label distribution:")
for label, count in Counter(predictions).most_common():
    print(f"  {label:<35} {count:>4} ({count/len(predictions)*100:.1f}%)")

print("\nReference label distribution (from CSV -1/0/1):")
for label, count in Counter(refs).most_common():
    print(f"  {label:<35} {count:>4} ({count/len(refs)*100:.1f}%)")

print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)
df_out = pd.DataFrame({"text": texts, "ref_label": refs, "pred_label": predictions})
for _, row in df_out.head(15).iterrows():
    print(f"  [{row.ref_label:>8}] → {row.pred_label}")
    print(f"           \"{row.text[:80]}\"")
    print()

# Save full results
out_path = "professor_predictions.csv"
df_out.to_csv(out_path, index=False)
print(f"Full predictions saved to: {out_path}")
