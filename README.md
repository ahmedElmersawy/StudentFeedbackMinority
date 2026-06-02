# Feedback Atlas

> **AI-powered minority signal detection for student feedback — built at Purdue University**

[![CI](https://github.com/ahmedElmersawy/StudentFeedbackMinority/actions/workflows/deploy.yml/badge.svg)](https://github.com/ahmedElmersawy/StudentFeedbackMinority/actions/workflows/deploy.yml)

**🔗 Live Demo:** https://disclosure-lands-consensus-amount.trycloudflare.com

Upload a CSV of student feedback and get back:
- **Label classification** across 11 CATME or 24 professor dimensions
- **Minority pattern detection** — students facing language barriers, disabilities, financial hardship, and more
- **Mismatch detection** — where numeric ratings contradict written sentiment
- **Review queue** — low-confidence predictions flagged for human review
- **Per-label CSV exports** — download any label's rows directly

---

## What it does

Feedback Atlas runs a full ML pipeline on uploaded student feedback datasets:

```
Upload CSV → Clean & anonymize → Classify (distilroberta-base)
           → Embed (MiniLM-L6-v2) → Minority detection (IsolationForest + DBSCAN)
           → Mismatch detection → Summary dashboard
```

It supports two feedback modes, auto-detected from the CSV:

| Mode | Model | Labels |
|------|-------|--------|
| **Student → Student** | CATME peer/self feedback | 11 labels (Majority_Positive, Minority_Peer_Experience, Suggestion_To_Peer, Negative_*) |
| **Student → Professor** | Course evaluations (RMP + Purdue) | 24 labels (Teaching_*, Content_*, Exam_*, Lab_*, Support_*, Minority_Student_Experience) |

---

## Dashboard

The platform is built as a modern analytics dashboard with:

- **Overview** — KPI cards, label distribution chart, confidence donut, minority snapshot
- **Pipeline Monitor** — live stage tracker with speed (rows/sec) and ETA
- **Label Explorer** — sortable per-label table with per-label CSV download
- **Minority Patterns** — expandable flagged rows with category breakdown
- **Mismatch Detection** — HIGH_MISMATCH and REVERSE_MISMATCH split view
- **Review Workspace** — keyboard-driven review (Y/N/J/K) with corrections export
- **Export Center** — full dataset, minority-only, per-label, mismatch, JSON report

---

## Performance

| Dataset size | Before | After |
|---|---|---|
| 50,000 rows | ~23 min | **~3 min** |
| 10,000 rows | ~5 min | **~45 sec** |

Key optimizations:
- Models cached in `/tmp` (local SSD) — eliminates NFS load latency
- FP16 inference + GPU warmup → classifier **4,400 rows/sec** on A30
- GPU nearest-neighbor via `torch.cdist` (replaces sklearn ball_tree)
- Server-Sent Events for real-time progress (no polling)
- DBSCAN on 2,000-row sample with GPU assignment to full dataset

---

## Models

| Model | Directory | Size |
|-------|-----------|------|
| CATME (11 labels) | `catme_feedback_classifier/` | 314 MB |
| Professor (24 labels) | `professor_feedback_classifier/` | 314 MB |

Train with SLURM on Purdue Gilbreth (A100-80GB):

```bash
sbatch slurm/train_catme.sh       # ~6h, zero-shot labels + fine-tune
sbatch slurm/train_professor.sh   # RMP + courseeval training
```

---

## Running locally

```bash
# 1. Clone
git clone https://github.com/ahmedElmersawy/StudentFeedbackMinority
cd StudentFeedbackMinority

# 2. Start backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 3. Start frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**

> **HPC note:** If running on a cluster, copy models to local disk first to avoid NFS latency:
> ```bash
> cp -r catme_feedback_classifier /tmp/fa_models/
> cp -r professor_feedback_classifier /tmp/fa_models/
> ```
> The backend auto-detects `/tmp/fa_models/` and prefers it over NFS.

---

## Docker (production)

```bash
docker compose up --build
# App: http://localhost  |  API docs: http://localhost:8000/docs
```

GPU passthrough is configured in `docker-compose.yml` automatically.

---

## Project structure

```
├── backend/                  FastAPI backend
│   ├── main.py               REST API + SSE streaming
│   ├── pipeline.py           Full ML pipeline (ingest → classify → detect)
│   ├── minority_detector.py  IsolationForest + DBSCAN minority detection
│   ├── mismatch_detector.py  Rating vs sentiment mismatch
│   ├── anonymizer.py         Name redaction
│   └── config.yaml           Labels, thresholds, zero-shot candidates
├── frontend/                 React + TypeScript + Vite dashboard
│   └── src/
│       ├── views/            Dashboard, LabelExplorer, ExportCenter, etc.
│       └── components/       Charts, ResultsTable, MinorityPanel, etc.
├── training/                 Training scripts
│   ├── train.py              Dual-mode trainer (CATME + professor)
│   └── data_loaders.py       CATME, RMP, courseeval loaders
├── slurm/                    SLURM job scripts for Purdue Gilbreth
├── scripts/                  ONNX conversion for CPU-only deployments
├── nginx/                    Production nginx config (SSE-aware)
└── .github/workflows/        CI/CD pipeline
```

---

## Deployment

See [DEPLOY.md](DEPLOY.md) for step-by-step instructions for:
- **Fly.io** (recommended — one-command deploy)
- **Railway** (simplest — GitHub auto-deploy)
- **Render** (free tier)

CI/CD is configured via GitHub Actions — set repo variable `DEPLOY_TARGET=fly` to activate auto-deploy on push.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, PyTorch 2.8, HuggingFace Transformers |
| Models | `distilroberta-base` fine-tuned, `all-MiniLM-L6-v2` embeddings |
| Minority detection | IsolationForest + DBSCAN (scikit-learn), GPU `torch.cdist` |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| Real-time | Server-Sent Events (SSE) with 10s keepalive |
| Deployment | Docker, nginx, Fly.io / Railway |

---

*Developed at Purdue University — Department of Computer Science*
