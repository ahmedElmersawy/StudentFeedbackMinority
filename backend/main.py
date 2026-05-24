"""FastAPI backend for Feedback Atlas."""
from __future__ import annotations

import io
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Configure logging
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
    description="Sentiment classification, minority detection, and mismatch analysis for student feedback.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store for async jobs (production should use Redis/DB)
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str
    model_dir: str | None = None


class PredictResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    needs_review: bool
    broad_sentiment: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | error
    message: str
    rows_processed: int = 0
    total_rows: int = 0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# /predict — single text
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Classify a single text string."""
    try:
        from .pipeline import run_inference

        df = pd.DataFrame({"text": [req.text]})
        result = run_inference(df, model_dir=req.model_dir)
        row = result.iloc[0]
        pred = str(row["prediction"])
        broad = _broad_sentiment(pred)
        return PredictResponse(
            text=req.text,
            prediction=pred,
            confidence=float(row["confidence"]),
            needs_review=bool(row["needs_review"]),
            broad_sentiment=broad,
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
    model_dir: str | None = Form(None),
    anonymize: bool = Form(True),
    include_minority: bool = Form(True),
    include_mismatch: bool = Form(True),
    run_zero_shot_categorization: bool = Form(False),
    confidence_threshold: float = Form(0.65),
):
    """
    Accept a CSV file, run the full pipeline asynchronously.
    Returns a job_id; poll /jobs/{job_id} for status and /jobs/{job_id}/results for data.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "message": "Queued",
        "rows_processed": 0,
        "total_rows": 0,
        "result_df": None,
        "error": None,
    }

    background_tasks.add_task(
        _run_pipeline_job,
        job_id=job_id,
        contents=contents,
        model_dir=model_dir,
        anonymize=anonymize,
        include_minority=include_minority,
        include_mismatch=include_mismatch,
        run_zero_shot_categorization=run_zero_shot_categorization,
        confidence_threshold=confidence_threshold,
    )

    return {"job_id": job_id, "status": "pending"}


def _run_pipeline_job(
    job_id: str,
    contents: bytes,
    **kwargs,
):
    """Background task: run full pipeline and store result."""
    from .pipeline import run_full_pipeline

    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["message"] = "Running pipeline…"
    try:
        df = run_full_pipeline(contents, **kwargs)
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["message"] = f"Complete — {len(df)} rows processed."
        _jobs[job_id]["rows_processed"] = len(df)
        _jobs[job_id]["total_rows"] = len(df)
        _jobs[job_id]["result_df"] = df
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
    )


@app.get("/jobs/{job_id}/results")
def job_results(job_id: str, format: str = "json"):
    """
    Retrieve pipeline results once job is done.
    ``format`` = ``json`` (default) or ``csv``.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job is not done yet (status: {job['status']}).")

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

    # JSON response — split into overview + row-level data
    return JSONResponse(_df_to_response(df))


# ---------------------------------------------------------------------------
# /minority — minority detection on uploaded CSV
# ---------------------------------------------------------------------------

@app.post("/minority")
async def minority(
    file: UploadFile = File(...),
    anonymize: bool = Form(True),
    run_zero_shot_categorization: bool = Form(False),
):
    """Accept a CSV, run minority detection only, return flagged rows."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    try:
        from .pipeline import ingest_csv
        from .minority_detector import detect_minority_patterns

        df, _, _, _ = ingest_csv(contents, anonymize=anonymize)
        df = df[df["text"].str.len() > 5].reset_index(drop=True)

        from backend.pipeline import _cfg
        cfg = _cfg()
        md_cfg = cfg.get("minority_detection", {})

        result = detect_minority_patterns(
            df,
            df["text"].tolist(),
            embedding_model=cfg.get("embeddings", {}).get("model", "all-MiniLM-L6-v2"),
            contamination=md_cfg.get("contamination", 0.08),
            min_cluster_size=md_cfg.get("min_cluster_size", 10),
            categorize=True,
            run_zero_shot_categorization=run_zero_shot_categorization,
        )

        flagged = result[result["is_minority_pattern"]].copy()
        return JSONResponse({
            "total": len(result),
            "flagged": len(flagged),
            "pct": round(100.0 * len(flagged) / max(1, len(result)), 2),
            "rows": flagged[["text", "is_outlier", "is_minority_cluster", "minority_category", "cluster_id"]]
                .rename(columns={"text": "feedback"})
                .fillna("")
                .to_dict(orient="records"),
        })
    except Exception as exc:
        logger.exception("Minority detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _broad_sentiment(label: str) -> str:
    l = label.lower()
    if l.startswith("positive"):
        return "Positive"
    if l.startswith("negative"):
        return "Negative"
    return "Neutral"


def _df_to_response(df: pd.DataFrame) -> dict:
    """Serialize a result DataFrame to a structured API response."""
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    total = len(df)
    minority_col = "is_minority_pattern"
    review_col = "needs_review"
    mismatch_col = "mismatch_flag"

    n_minority = int(df[minority_col].sum()) if minority_col in df.columns else 0
    n_review = int(df[review_col].sum()) if review_col in df.columns else 0
    n_mismatch = int(df[mismatch_col].sum()) if mismatch_col in df.columns else 0

    # Label distribution
    label_dist: dict[str, int] = {}
    if "prediction" in df.columns:
        label_dist = df["prediction"].value_counts().to_dict()

    # Category breakdown (minority)
    cat_breakdown: dict[str, int] = {}
    if "minority_category" in df.columns:
        from .minority_detector import category_breakdown
        flagged_df = df[df.get(minority_col, pd.Series([False] * total, dtype=bool)).astype(bool)]
        cat_breakdown = category_breakdown(flagged_df)

    # Avg confidence
    avg_conf = float(df["confidence"].mean()) if "confidence" in df.columns else 0.0

    return {
        "summary": {
            "total": total,
            "minority": n_minority,
            "minority_pct": round(100.0 * n_minority / max(1, total), 2),
            "needs_review": n_review,
            "review_pct": round(100.0 * n_review / max(1, total), 2),
            "mismatch": n_mismatch,
            "mismatch_pct": round(100.0 * n_mismatch / max(1, total), 2),
            "avg_confidence": round(avg_conf, 4),
            "label_distribution": label_dist,
            "minority_category_breakdown": cat_breakdown,
        },
        "rows": df.fillna("").to_dict(orient="records"),
    }
