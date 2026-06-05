"""FastAPI backend for Feedback Atlas — Dual-Mode."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
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
# Rate limiting — simple in-memory token bucket per IP
# ---------------------------------------------------------------------------
_MAX_FILE_MB  = int(os.environ.get("MAX_UPLOAD_MB", "500"))   # 500 MB default
_RATE_WINDOW  = 60          # seconds per window
_RATE_MAX     = 5           # max uploads per window per IP
_rate_log: dict[str, list[float]] = defaultdict(list)

def _check_rate(request: Request) -> None:
    """Allow MAX uploads per IP per 60 s. Raises 429 if exceeded."""
    ip  = request.client.host if request.client else "unknown"
    now = time.time()
    hits = [t for t in _rate_log[ip] if now - t < _RATE_WINDOW]
    if len(hits) >= _RATE_MAX:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit: max {_RATE_MAX} uploads per {_RATE_WINDOW}s. Try again shortly.",
        )
    hits.append(now)
    _rate_log[ip] = hits


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
    request: Request,
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
    _check_rate(request)   # 429 if IP exceeds rate limit

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > _MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(contents)//1024//1024} MB). Max allowed: {_MAX_FILE_MB} MB.",
        )

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

    import time as _time
    _jobs[job_id]["status"] = "running"
    _jobs[job_id]["message"] = "Loading model…"
    _jobs[job_id]["speed_rows_per_sec"] = 0.0
    _jobs[job_id]["stage"] = "loading"

    def _progress(done: int, total: int, speed: float = 0.0) -> None:
        remaining = int((total - done) / speed) if speed > 0 and done < total else 0
        _jobs[job_id]["rows_processed"]    = done
        _jobs[job_id]["total_rows"]        = total
        _jobs[job_id]["speed_rows_per_sec"]= round(speed, 1)
        _jobs[job_id]["stage"]             = "classifying"
        _jobs[job_id]["message"] = (
            f"Classifying… {done:,}/{total:,} rows"
            + (f" · {speed:,.0f} rows/sec" if speed > 0 else "")
            + (f" · {remaining}s remaining" if remaining > 0 and remaining < 3600 else "")
        )

    try:
        df = run_full_pipeline(contents, feedback_mode=feedback_mode,
                               progress_callback=_progress, **kwargs)
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


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    """
    Server-Sent Events stream for real-time pipeline progress.
    Replaces polling — client receives updates every 400 ms.
    """
    async def _gen():
        last_msg    = None
        ping_ticks  = 0        # send `: ping` comment every 25 s to beat proxy timeouts

        while True:
            job = _jobs.get(job_id)
            if not job:
                yield f"event: error\ndata: {json.dumps({'message': 'Job not found'})}\n\n"
                return

            payload = {
                "status":             job["status"],
                "message":            job["message"],
                "rows_processed":     job["rows_processed"],
                "total_rows":         job["total_rows"],
                "feedback_mode":      job.get("feedback_mode"),
                "speed_rows_per_sec": job.get("speed_rows_per_sec", 0),
                "stage":              job.get("stage", ""),
            }
            msg = json.dumps(payload)
            if msg != last_msg:
                yield f"data: {msg}\n\n"
                last_msg = msg

            if job["status"] in ("done", "error"):
                return

            # Keepalive ping every ~10 s — Cloudflare kills idle SSE after ~30 s,
            # so we ping at 10 s to stay well under that threshold.
            ping_ticks += 1
            if ping_ticks % 25 == 0:   # 25 × 0.4 s = 10 s
                yield ": ping\n\n"

            await asyncio.sleep(0.4)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",        # disable nginx buffering
            "Connection":       "keep-alive",
        },
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

    # Cap JSON rows to avoid OOM / browser freeze on large datasets.
    # Full data is always available via ?format=csv.
    JSON_ROW_LIMIT = 10_000
    df_json = df.head(JSON_ROW_LIMIT) if len(df) > JSON_ROW_LIMIT else df
    return JSONResponse(_df_to_response(df_json, job, total_rows=len(df)))


@app.get("/jobs/{job_id}/results/minority")
def job_minority_results(job_id: str):
    """Download only minority-flagged rows as CSV."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (status: {job['status']}).")

    df: pd.DataFrame = job["result_df"]
    minority_col = "is_minority_pattern"
    if minority_col not in df.columns:
        raise HTTPException(status_code=404, detail="No minority detection data in this job.")

    minority_df = df[df[minority_col].astype(bool)].copy()
    buf = io.StringIO()
    minority_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=minority_results_{job_id}.csv"},
    )


@app.get("/jobs/{job_id}/labels")
def job_labels(job_id: str):
    """Return per-label stats: count, pct, avg_confidence for every prediction label."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (status: {job['status']}).")

    df: pd.DataFrame = job["result_df"]
    if "prediction" not in df.columns:
        return {"labels": []}

    total = len(df)
    result = []
    for label, group in df.groupby("prediction"):
        result.append({
            "label": str(label),
            "count": int(len(group)),
            "pct": round(100.0 * len(group) / max(1, total), 2),
            "avg_confidence": round(float(group["confidence"].mean()) if "confidence" in group.columns else 0.0, 4),
            "minority_count": int(group["is_minority_pattern"].astype(bool).sum()) if "is_minority_pattern" in group.columns else 0,
        })
    result.sort(key=lambda x: -x["count"])
    return _to_python({"job_id": job_id, "total": total, "labels": result})


@app.get("/jobs/{job_id}/results/label/{label}")
def job_label_results(job_id: str, label: str):
    """Download all rows for a specific prediction label as CSV."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (status: {job['status']}).")

    df: pd.DataFrame = job["result_df"]
    if "prediction" not in df.columns:
        raise HTTPException(status_code=404, detail="No prediction data in this job.")

    label_df = df[df["prediction"] == label].copy()
    if label_df.empty:
        raise HTTPException(status_code=404, detail=f"Label '{label}' not found in results.")

    safe_label = label.replace("/", "_").replace(" ", "_")
    buf = io.StringIO()
    label_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.read()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={safe_label}_{job_id}.csv"},
    )


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


@app.get("/jobs/{job_id}/aggregate")
def job_aggregate(job_id: str):
    """
    Return professor/dimension-level aggregated summary for a completed job.

    Each row = one professor, course, or feedback dimension with:
      - total_responses
      - flagged_findings: labels that ≥ N students mentioned (threshold in config)
      - needs_attention: true when at least one actionable finding meets threshold
      - top_negative_label, minority_count, avg_priority_score
      - label_counts: full label distribution

    This is the primary output for university administrators — it collapses
    individual comments into actionable patterns per entity.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (status: {job['status']}).")

    from .pipeline import aggregate_by_entity
    df: pd.DataFrame = job["result_df"]
    summary = aggregate_by_entity(df)

    if summary.empty:
        return {"job_id": job_id, "feedback_mode": job.get("feedback_mode"), "entities": []}

    return JSONResponse({
        "job_id": job_id,
        "feedback_mode": job.get("feedback_mode"),
        "entities": summary.to_dict(orient="records"),
    })


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


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python-native types for JSON."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _df_to_response(df: pd.DataFrame, job: dict | None = None, total_rows: int | None = None) -> dict:
    """Serialize a result DataFrame to a structured API response.
    total_rows overrides len(df) for summary stats when df is a truncated slice."""
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)

    total = total_rows if total_rows is not None else len(df)
    minority_col = "is_minority_pattern"
    review_col = "needs_review"
    mismatch_col = "mismatch_flag"
    priority_col = "priority_score"

    # Use the full DataFrame from the job for summary stats when a truncated slice is passed.
    full_df = job["result_df"] if (job and total_rows is not None and "result_df" in job) else df

    n_minority = int(full_df[minority_col].sum()) if minority_col in full_df.columns else 0
    n_review = int(full_df[review_col].sum()) if review_col in full_df.columns else 0
    n_mismatch = int(full_df[mismatch_col].sum()) if mismatch_col in full_df.columns else 0
    avg_priority = float(full_df[priority_col].mean()) if priority_col in full_df.columns else 0.0

    label_dist: dict[str, int] = {}
    if "prediction" in full_df.columns:
        label_dist = full_df["prediction"].value_counts().to_dict()

    cat_breakdown: dict[str, int] = {}
    if "minority_category" in full_df.columns:
        from .minority_detector import category_breakdown
        flagged_df = full_df[full_df[minority_col].astype(bool)] if minority_col in full_df.columns else pd.DataFrame()
        cat_breakdown = category_breakdown(flagged_df)

    # CATME subtype breakdown
    subtype_dist: dict[str, int] = {}
    if "catme_subtype" in full_df.columns:
        subtype_dist = full_df["catme_subtype"].value_counts().to_dict()

    avg_conf = float(full_df["confidence"].mean()) if "confidence" in full_df.columns else 0.0
    feedback_mode = str(full_df["feedback_mode"].iloc[0]) if "feedback_mode" in full_df.columns else "unknown"

    return _to_python({
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
            "rows_truncated": total_rows is not None and len(df) < total,
        },
        "rows": df.fillna("").to_dict(orient="records"),
    })
