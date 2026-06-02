# Feedback Atlas — Deployment Guide

## Quick start (Docker, any Linux server with NVIDIA GPU)

```bash
git clone https://github.com/ahmedElmersawy/StudentFeedbackMinority
cd StudentFeedbackMinority

# Models must exist (train first or copy from scratch)
# ls catme_feedback_classifier/model.safetensors
# ls professor_feedback_classifier/model.safetensors

docker compose up --build -d
```

- **App:** http://localhost (nginx → React SPA)
- **API:** http://localhost:8000 (FastAPI)
- **Docs:** http://localhost:8000/docs

---

## Option A — Fly.io (recommended, global edge, free tier available)

### 1. Install flyctl
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Launch the backend app
```bash
fly launch --name feedback-atlas-api --region ord --no-deploy
# Edit fly.toml if needed (see file in repo root)

# Create a persistent volume for model weights (one-time)
fly volumes create model_weights --region ord --size 20

# Upload model weights (after training on SLURM)
fly ssh sftp shell
# then: put catme_feedback_classifier/model.safetensors /app/models/catme/model.safetensors

fly deploy
```

Your API is live at: **https://feedback-atlas-api.fly.dev**

### 3. Deploy the frontend to Vercel
```bash
npm install -g vercel
cd frontend
VITE_API_URL=https://feedback-atlas-api.fly.dev npm run build
vercel --prod
```

Vercel gives you: **https://feedback-atlas-xyz.vercel.app**

---

## Option B — Railway (simplest, one-click deploy)

1. Fork this repo on GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select your fork
4. Set environment variables:
   ```
   TOKENIZERS_PARALLELISM=false
   ```
5. Add a **Volume** → mount at `/app/models`
6. Upload model weights via Railway's file browser or SSH

Railway gives you: **https://feedback-atlas.up.railway.app**

---

## Option C — Render (free tier, auto-deploys from GitHub)

1. New Web Service → connect GitHub repo
2. Build command: `pip install -r backend/requirements.txt`
3. Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add a **Disk** at `/app/models`, 20 GB
5. Environment variables:
   ```
   TOKENIZERS_PARALLELISM=false
   PYTHON_VERSION=3.11
   ```

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `CORS_ORIGINS` | No | Comma-separated allowed origins (default: `*`) |
| `TOKENIZERS_PARALLELISM` | Yes | Set to `false` to avoid HuggingFace warnings |
| `VITE_API_URL` | Frontend build | Full backend URL (empty = nginx proxy) |

---

## CI/CD (GitHub Actions)

The workflow in `.github/workflows/deploy.yml` auto-deploys on push to `main`.

### Required secrets (GitHub → Settings → Secrets):

| Secret | Used by |
|---|---|
| `FLY_API_TOKEN` | Fly.io deploy |
| `RAILWAY_TOKEN` | Railway deploy |
| `VERCEL_TOKEN` | Vercel frontend deploy |
| `VERCEL_ORG_ID` | Vercel |
| `VERCEL_PROJECT_ID` | Vercel |
| `VITE_API_URL` | Frontend build (e.g. `https://feedback-atlas-api.fly.dev`) |

### Required variable (GitHub → Settings → Variables):

| Variable | Value |
|---|---|
| `DEPLOY_TARGET` | `fly`, `railway`, or `vercel` |

---

## Performance guide

### With NVIDIA GPU (A30, V100, A100)

| Rows | Time (before) | Time (after) |
|---|---|---|
| 10,000 | ~5 min | **~45 sec** |
| 50,000 | ~23 min | **~3 min** |
| 200,000 | ~90 min | **~10 min** |

Speedups achieved:
- FP16 inference + GPU warmup: classifier 23 rows/sec → **4,400 rows/sec**
- Embedding batch 512 → 2048 + GPU: 140 rows/sec → **~600 rows/sec**
- SSE streaming replaces polling (real-time, no client-side overhead)
- Model cached across requests (zero reload time after first job)

### Without GPU (CPU-only deployment)

Use ONNX Runtime for 2–4× CPU speedup:
```bash
pip install onnxruntime
python scripts/convert_to_onnx.py --model catme_feedback_classifier
python scripts/convert_to_onnx.py --model professor_feedback_classifier
```

| Rows | PyTorch CPU | ONNX CPU |
|---|---|---|
| 50,000 | ~23 min | **~8 min** |

---

## Monitoring

### Health check
```bash
curl https://your-app.fly.dev/health
# → {"status":"ok","version":"3.0.0"}
```

### Log streaming (Fly.io)
```bash
fly logs --app feedback-atlas-api
```

### Metrics
- Fly.io: built-in metrics at fly.io/apps/feedback-atlas-api/metrics
- Railway: built-in logging dashboard
- Self-hosted: `docker stats`

---

## Security checklist

- [ ] Set `CORS_ORIGINS` to your frontend domain (not `*`) in production
- [ ] Place models behind a private volume, not public object storage
- [ ] Enable HTTPS (Fly.io/Railway/Vercel all do this automatically)
- [ ] Rotate API tokens after first deploy
- [ ] Set `client_max_body_size 100M` in nginx if you want to cap upload size

---

## Reproducing the full stack locally

```bash
# 1. Train models (requires SLURM / GPU)
sbatch slurm/train_catme.sh
sbatch slurm/train_professor.sh

# 2. Start dev servers
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
cd frontend && npm run dev

# 3. Open http://localhost:5173
```

Or with Docker:
```bash
docker compose up --build
# Open http://localhost
```
