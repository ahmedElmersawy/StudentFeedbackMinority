"""
Convert a trained distilroberta classifier to ONNX format for fast CPU inference.

ONNX Runtime is 2–4× faster than PyTorch on CPU for inference-only workloads.
No GPU required at runtime.

Usage:
    pip install onnxruntime onnx optimum
    python scripts/convert_to_onnx.py --model catme_feedback_classifier
    python scripts/convert_to_onnx.py --model professor_feedback_classifier

After conversion, the backend auto-detects model.onnx and uses ONNX Runtime
instead of PyTorch when CUDA is not available.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def convert(model_dir: Path, output_path: Path | None = None) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}")

    out = output_path or model_dir / "model.onnx"
    print(f"Loading {model_dir} …")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    # Dummy input — max_length=128 covers most CATME/courseeval texts
    dummy = tokenizer(
        ["This is a sample feedback text for ONNX export."],
        return_tensors="pt",
        padding="max_length",
        max_length=128,
        truncation=True,
    )

    print(f"Exporting to ONNX: {out} …")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits":         {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    # Verify
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    out_ort = sess.run(None, {
        "input_ids":      dummy["input_ids"].numpy(),
        "attention_mask": dummy["attention_mask"].numpy(),
    })
    print(f"✓ ONNX verified — output shape: {out_ort[0].shape}")

    # Write conversion metadata
    meta_path = model_dir / "onnx_metadata.json"
    meta_path.write_text(json.dumps({
        "onnx_path": str(out),
        "max_length": 128,
        "opset": 17,
    }, indent=2))

    print(f"\nDone. Model saved to: {out}")
    print("The backend will use ONNX automatically when CUDA is unavailable.")
    return out


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace model to ONNX.")
    parser.add_argument("--model", required=True, help="Model directory name or path")
    parser.add_argument("--output", default=None, help="Output .onnx file path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    mdir = Path(args.model) if Path(args.model).is_absolute() else root / args.model
    out  = Path(args.output) if args.output else None
    convert(mdir, out)


if __name__ == "__main__":
    main()
