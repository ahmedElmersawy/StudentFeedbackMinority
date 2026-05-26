"""FastAPI backend for Feedback Atlas — Dual-Mode."""
from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Feedback Atlas API",
    description=(
        "Dual-mode sentiment classification, minority detection, and mismatch analysis "
        "for student feedback. Mode: student_to_student (CATME) | student_to_professor (RMP/courseeval)."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, dict[str, Any]] = {}

# Split output files are stored here while the job is alive
_OUTPUT_DIR = Path("/tmp/feedback_atlas_outputs")
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str
    model_dir: Optional[str] = None
    feedback_mode: Optional[str] = None


class PredictResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    needs_review: bool
    broad_sentiment: str
    priority_score: int


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | error
    message: str
    rows_processed: int = 0
    total_rows: int = 0
    feedback_mode: Optional[str] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


# ---------------------------------------------------------------------------
# /predict — single text
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Classify a single text string."""
    try:
        from .pipeline import run_inference, calculate_priority, detect_feedback_mode, _cfg

        df = pd.DataFrame({"text": [req.text]})
        mode = req.feedback_mode or detect_feedback_mode([req.text])
        result = run_inference(df, model_dir=req.model_dir)
        row = result.iloc[0]
        pred = str(row["prediction"])
        conf = float(row["confidence"])
        prio = calculate_priority(pred, False, ["Statistical_Outlier_Only"], conf, _cfg())
        return PredictResponse(
            text=req.text,
            prediction=pred,
            confidence=conf,
            needs_review=bool(row["needs_review"]),
            broad_sentiment=_broad_sentiment(pred),
            priority_score=prio,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not found: {exc}") from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# /upload — full pipeline (async)
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    feedback_mode: Optional[str] = Form(None),
    model_dir: Optional[str] = Form(None),
    anonymize: bool = Form(True),
    include_minority: bool = Form(True),
    include_mismatch: bool = Form(True),
    run_zero_shot_categorization: bool = Form(False),
    confidence_threshold: float = Form(0.65),
    generate_split_outputs: bool = Form(False),
):
    """
    Accept a CSV file, run the full pipeline asynchronously.
    feedback_mode: 'student_to_student' | 'student_to_professor' | None (auto-detect).
    generate_split_outputs: if True, write per-mode split CSVs to /tmp.
    Returns job_id; poll /jobs/{job_id} for status.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if feedback_mode and feedback_mode not in ("student_to_student", "student_to_professor"):
        raise HTTPException(
            status_code=400,
            detail="feedback_mode must be 'student_to_student', 'student_to_professor', or omitted for auto-detect.",
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "message": "Queued",
        "rows_processed": 0,
        "total_rows": 0,
        "result_df": None,
        "error": None,
        "feedback_mode": feedback_mode,
        "success_criteria": None,
        "split_files": {},
    }

    background_tasks.add_task(
        _run_pipeline_job,
        job_id=job_id,
        contents=contents,
        feedback_mode=feedback_mode,
        model_dir=model_dir,
        anonymize=anonymize,
        include_minority=include_minority,
        include_mismatch=include_mismatch,
        run_zero_shot_categorization=run_zero_shot_categorization,
        confidence_threshold=confidence_threshold,
        generate_split_outputs=generate_split_outputs,
    )

    return {"job_id": job_id, "status": "pending"}


def _run_pipeline_job(
    job_id: str,
    contents: bytes,
    feedback_mode: Optional[str],
    generate_split_outputs: bool,
    **kwargs,
):
    """Background task: run full pipeline and store result."""
    from .pipeline import run_full_pipeline, generate_output_files, check_success_criteria

    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["message"] = "Running pipeline…"
    try:
        df = run_full_pipeline(contents, feedback_mode=feedback_mode, **kwargs)
        detected_mode = str(df["feedback_mode"].iloc[0]) if "feedback_mode" in df.columns else (feedback_mode or "unknown")

        success = check_success_criteria(df, detected_mode)
        split_files: dict[str, str] = {}

        if generate_split_outputs:
            job_out_dir = _OUTPUT_DIR / job_id
            written = generate_output_files(df, job_out_dir, detected_mode)
            split_files = {name: str(path) for name, path in written.items()}

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["message"] = f"Complete — {len(df)} rows processed."
        _jobs[job_id]["rows_processed"] = len(df)
        _jobs[job_id]["total_rows"] = len(df)
        _jobs[job_id]["result_df"] = df
        _jobs[job_id]["feedback_mode"] = detected_mode
        _jobs[job_id]["success_criteria"] = success
        _jobs[job_id]["split_files"] = split_files
    except Exception as exc:
        logger.exception("Pipeline job %s failed", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["message"] = str(exc)
        _jobs[job_id]["error"] = str(exc)


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    """Poll pipeline job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job["message"],
        rows_processed=job["rows_processed"],
        total_rows=job["total_rows"],
        feedback_mode=job.get("feedback_mode"),
    )


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str, format: str = "json"):
    """
    Retrieve pipeline results once job is done.
    format = json (default) or csv.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (status: {job['status']}).")

    df: pd.DataFrame = job["result_df"]

    if format == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(
            iter([buf.read()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=results_{job_id}.csv"},
        )

    return JSONResponse(_df_to_response(df, job))


@app.get("/jobs/{job_id}/download/{filename}")
def download_split_file(job_id: str, filename: str):
    """
    Download a specific split output file (e.g. priority_alerts.csv).
    Only available when generate_split_outputs=True was passed to /upload.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="Job not done yet.")

    split_files: dict[str, str] = job.get("split_files", {})
    if filename not in split_files:
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found. Available: {list(split_files.keys())}")

    path = Path(split_files[filename])
    if not path.exists():
        raise HTTPException(status_code=410, detail="File was deleted or never written.")

    with open(path, "rb") as f:
        content = f.read()
    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/jobs/{job_id}/files")
def list_split_files(job_id: str):
    """List available split output files for a completed job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "feedback_mode": job.get("feedback_mode"),
        "files": list(job.get("split_files", {}).keys()),
        "success_criteria": job.get("success_criteria"),
    }


# ---------------------------------------------------------------------------
# /minority — minority detection only
# ---------------------------------------------------------------------------

@app.post("/minority")
async def minority(
    file: UploadFile = File(...),
    anonymize: bool = Form(True),
    run_zero_shot_categorization: bool = Form(False),
    feedback_mode: Optional[str] = Form(None),
):
    """Accept a CSV, run minority detection only, return flagged rows."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        from .pipeline import ingest_csv, detect_feedback_mode, keyword_minority_detection, _cfg
        from .minority_detector import detect_minority_patterns

        df, _, _, _ = ingest_csv(contents, anonymize=anonymize)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)

        if feedback_mode is None:
            feedback_mode = detect_feedback_mode(df["text"].dropna().tolist())

        cfg = _cfg()
        md_cfg = cfg.get("minority_detection", {})
        keyword_map = cfg.get("minority_keywords", {})

        # Keyword pass
        kw_is_minority: list[bool] = []
        kw_categories: list[str] = []
        for text in df["text"].tolist():
            is_min, cats = keyword_minority_detection(text, keyword_map)
            kw_is_minority.append(is_min)
            kw_categories.append("|".join(cats))
        df["keyword_minority"] = kw_is_minority
        df["keyword_categories"] = kw_categories

        # Embedding pass
        result = detect_minority_patterns(
            df,
            df["text"].tolist(),
            embedding_model=cfg.get("embeddings", {}).get("model", "all-MiniLM-L6-v2"),
            contamination=md_cfg.get("contamination", 0.08),
            min_cluster_size=md_cfg.get("min_cluster_size", 10),
            categorize=True,
            run_zero_shot_categorization=run_zero_shot_categorization,
        )

        result["is_minority_pattern"] = result["is_minority_pattern"] | result["keyword_minority"]
        result["minority_category"] = result.apply(
            lambda row: row["keyword_categories"] if row["keyword_minority"]
            else (row.get("minority_category", "") or ""),
            axis=1,
        )

        flagged = result[result["is_minority_pattern"]].copy()
        return JSONResponse({
            "total": len(result),
            "flagged": len(flagged),
            "pct": round(100.0 * len(flagged) / max(1, len(result)), 2),
            "feedback_mode": feedback_mode,
            "rows": flagged[
                ["text", "is_outlier", "is_minority_cluster", "minority_category", "cluster_id"]
            ].rename(columns={"text": "feedback"}).fillna("").to_dict(orient="records"),
        })
    except Exception as exc:
        logger.exception("Minority detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broad_sentiment(label: str) -> str:
    l = label.lower()
    if any(l.startswith(p) for p in ("positive_", "teaching_positive", "content_positive",
                                      "exam_positive", "lab_positive", "support_positive",
                                      "majority_positive", "self_positive")):
        return "Positive"
    if any(l.startswith(p) for p in ("negative_", "teaching_negative", "content_negative",
                                      "exam_negative", "lab_negative", "support_negative",
                                      "self_struggle", "self_minority")):
        return "Negative"
    return "Neutral"


def _df_to_response(df: pd.DataFrame, job: dict | None = None) -> dict:
    """Serialize a result DataFrame to a structured API response."""
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    total = len(df)
    minority_col = "is_minority_pattern"
    review_col = "needs_review"
    mismatch_col = "mismatch_flag"
    priority_col = "priority_score"

    n_minority = int(df[minority_col].sum()) if minority_col in df.columns else 0
    n_review = int(df[review_col].sum()) if review_col in df.columns else 0
    n_mismatch = int(df[mismatch_col].sum()) if mismatch_col in df.columns else 0
    avg_priority = float(df[priority_col].mean()) if priority_col in df.columns else 0.0

    label_dist: dict[str, int] = {}
    if "prediction" in df.columns:
        label_dist = df["prediction"].value_counts().to_dict()

    cat_breakdown: dict[str, int] = {}
    if "minority_category" in df.columns:
        from .minority_detector import category_breakdown
        flagged_df = df[df[minority_col].astype(bool)] if minority_col in df.columns else pd.DataFrame()
        cat_breakdown = category_breakdown(flagged_df)

    # CATME subtype breakdown
    subtype_dist: dict[str, int] = {}
    if "catme_subtype" in df.columns:
        subtype_dist = df["catme_subtype"].value_counts().to_dict()

    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0
    feedback_mode = str(df["feedback_mode"].iloc[0]) if "feedback_mode" in df.columns else "unknown"

    return {
        "summary": {
            "total": total,
            "feedback_mode": feedback_mode,
            "minority": n_minority,
            "minority_pct": round(100.0 * n_minority / max(1, total), 2),
            "needs_review": n_review,
            "review_pct": round(100.0 * n_review / max(1, total), 2),
            "mismatch": n_mismatch,
            "mismatch_pct": round(100.0 * n_mismatch / max(1, total), 2),
            "avg_confidence": round(avg_conf, 4),
            "avg_priority_score": round(avg_priority, 2),
            "label_distribution": label_dist,
            "minority_category_breakdown": cat_breakdown,
            "catme_subtype_distribution": subtype_dist,
            "success_criteria": job.get("success_criteria") if job else None,
            "split_files": list(job.get("split_files", {}).keys()) if job else [],
        },
        "rows": df.fillna("").to_dict(orient="records"),
    }
