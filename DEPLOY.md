# Deploy so anyone can open the app

You have **three** entry points in this repo:

| App | Command | Best for |
|-----|-----------|----------|
| **Reflex** (newer UI) | `reflex run` | Custom layout, charts, public site |
| **Streamlit** | `py -3.12 -m streamlit run app.py` | Fastest deploy on Streamlit Community Cloud |
| **React SPA** | `cd frontend && npm run dev` | Class demo UI (mock detection); host `npm run build` on Netlify / static host |

See `frontend/README.md` for the TypeScript app.

---

## Reflex (recommended UI)

### Why it “only worked on localhost”

Browsers must reach **two** services: the **frontend** (HTML/JS) and the **backend** (WebSocket + uploads). For a public URL you must set **`API_URL`** to the **public backend address** (usually port **8000**), then rebuild or recompile so the frontend embeds that URL.

Example:

```bash
export API_URL=https://feedback-api.yourdomain.com:8000
reflex run --env prod --backend-host 0.0.0.0
```

Or bake it at **image build** time:

```bash
docker build --build-arg API_URL=https://feedback-api.yourdomain.com:8000 -t feedback-web .
```

### Model path on a server

The UI field `final_feedback_classifier` is resolved against the **current working directory** and the repo root. On a server, set an absolute path:

```bash
export FEEDBACK_MODEL_DIR=/data/models/final_feedback_classifier
```

Or mount your model folder into the container and point `FEEDBACK_MODEL_DIR` there.

### Docker (VPS, Fly.io, EC2, etc.)

```bash
docker build -t feedback-web .
docker run -d -p 3000:3000 -p 8000:8000 \
  -e API_URL=http://YOUR_PUBLIC_IP:8000 \
  -e FEEDBACK_MODEL_DIR=/models/final_feedback_classifier \
  -v /path/on/host/models:/models:ro \
  --name feedback feedback-web
```

Put **Nginx** or **Caddy** in front with TLS. Enable **WebSocket** proxying to port 8000 (required for Reflex state).

### Free / simple hosts

- **Streamlit Cloud** does **not** run Reflex; use the Streamlit section below for one-click hosting of `app.py`.
- **Reflex Cloud**: see [reflex.dev](https://reflex.dev) hosting docs (`reflex deploy`).
- **Railway / Render**: use this `Dockerfile`, set `API_URL` to your service’s public backend URL, expose ports **3000** and **8000** (or terminate TLS at the proxy).

---

## Streamlit (original `app.py`)

### Streamlit Community Cloud (free tier)

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app with main file `app.py`.
3. Set Python **3.12** and `requirements.txt`.

#### “That folder path does not exist” / model missing on Cloud

The default sidebar path `final_feedback_classifier` is the folder **next to** `app.py` after training. That directory is **gitignored** in this repo (large weights), so **Streamlit Cloud clones a repo that does not contain the model** unless you add it.

**Ways to fix it:**

| Approach | What to do |
|----------|------------|
| **A. Ship the model in Git** | Remove `final_feedback_classifier/` from `.gitignore` (or only ignore checkpoints), commit the folder, push. Use **[Git LFS](https://git-lfs.github.com/)** for `*.safetensors` / large shards so the repo stays usable. |
| **B. Env var on a custom host** | If you run Streamlit on a VM/Docker with a volume mount, set `FEEDBACK_MODEL_DIR` to the **absolute** path of the classifier folder. The app auto-loads from this variable when it is set. |
| **C. Run locally** | Train or copy `final_feedback_classifier/` beside `app.py`, then `streamlit run app.py`. |

Path resolution: relative names are searched under the **repository root** and the process **working directory**; `FEEDBACK_MODEL_DIR` overrides the sidebar when set.

---

### Step-by-step: ship `final_feedback_classifier/` with Git LFS (Streamlit Cloud)

Do this **once** on your PC, inside your clone of the repo (same folder as `app.py`).

**1. Install Git LFS**

- Windows: [Git LFS releases](https://git-lfs.github.com/) installer, or `winget install GitHub.GitLFS`, then reopen the terminal.
- macOS: `brew install git-lfs`
- Linux: `sudo apt install git-lfs` (or your package manager)

Then enable it for your user:

```bash
git lfs install
```

**2. Tell Git LFS which files inside the model folder are large**

Create or edit **`.gitattributes`** in the **repo root** (next to `app.py`) so weights are stored as LFS pointers, not giant normal Git blobs:

```gitattributes
# Hugging Face classifier weights (adjust if your export uses different names)
final_feedback_classifier/*.safetensors filter=lfs diff=lfs merge=lfs -text
final_feedback_classifier/**/*.safetensors filter=lfs diff=lfs merge=lfs -text
final_feedback_classifier/*.bin filter=lfs diff=lfs merge=lfs -text
```

If you only have `model.safetensors`, the first line is enough.

**3. Stop ignoring the model folder**

In **`.gitignore`**, delete or comment out this line (only the one for the folder you are shipping):

```gitignore
final_feedback_classifier/
```

Leave `final_feedback_classifier_v2/` ignored if you do not need it on Cloud.

**4. Add, commit, push**

```bash
git add .gitattributes .gitignore
git add final_feedback_classifier/
git status   # you should see "LFS" next to large files if LFS is active
git commit -m "Add classifier for Streamlit Cloud (Git LFS)"
git push origin main
```

**5. Redeploy Streamlit Cloud**

Open your app on [share.streamlit.io](https://share.streamlit.io) → **Reboot** or trigger a redeploy so it pulls the latest commit.

**Notes**

- GitHub gives a **free LFS quota** (storage + bandwidth). If `git push` complains about LFS, check [Billing → Plans](https://github.com/settings/billing) or add a **data pack**, or shrink the model / use fewer files.
- If you **already committed** `model.safetensors` as a normal (non-LFS) file, fix history with  
  `git lfs migrate import --include="*.safetensors" --everything`  
  (rewrites commits; coordinate if others use the repo.)

---

### Step-by-step: Docker / VPS with `FEEDBACK_MODEL_DIR`

Use this when the model lives **on the server disk** (or in a volume), not inside the Git clone.

**1. Put the folder on the host**

Example host path:

```text
/opt/models/final_feedback_classifier/
```

That directory must contain `config.json`, tokenizer files, and weights (`model.safetensors` or equivalent), same layout as after training.

**2. Run Streamlit with a mount + env**

Example **Docker** (Linux/macOS paths; on **Windows CMD** use `^` at line ends instead of `\`):

```bash
docker run --rm -p 8501:8501 \
  -v /opt/models/final_feedback_classifier:/models/classifier:ro \
  -e FEEDBACK_MODEL_DIR=/models/classifier \
  -w /app \
  python:3.12-bookworm \
  bash -lc "pip install --no-cache-dir -r requirements.txt && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
```

Mount your real host folder into `/models/classifier` (left side of `-v` is host, right side is inside the container). `FEEDBACK_MODEL_DIR` must match the **in-container** path.

**3. Behaviour**

With `FEEDBACK_MODEL_DIR` set, the Streamlit app **auto-loads** that folder on startup (no sidebar click needed), as long as the path exists inside the container and contains `config.json`.

**4. Reflex / Dockerfile in this repo**

The same variable works for Reflex: set `FEEDBACK_MODEL_DIR` in the container environment and mount the folder to the path you use. See the `docker run` example in the Reflex section above.

### Repo files added for you

| File | Purpose |
|------|---------|
| `.gitattributes` | Routes `*.safetensors` / `*.bin` under `final_feedback_classifier/` through **Git LFS**. |
| `Dockerfile` | Reflex image **bakes** `final_feedback_classifier/` and sets `FEEDBACK_MODEL_DIR=/app/final_feedback_classifier`. |
| `Dockerfile.streamlit` | Slim Streamlit image; expects a **volume** at `/models/classifier` by default. |
| `docker-compose.yml` | `docker compose up --build` → Streamlit on **:8501** with local folder mounted. Optional `reflex` service: `docker compose --profile reflex up --build` (set `API_URL` for public deploy). |
| `.env.example` | Copy to `.env` and fill `FEEDBACK_MODEL_DIR` / `API_URL` when needed. |

### Run on a server (any machine on your network)

Streamlit already prints a **Network URL** when you use `address = 0.0.0.0`. This repo includes `.streamlit/config.toml` with:

- `address = "0.0.0.0"` — listen on all interfaces  
- `headless = true` — suitable for servers  

Start with:

```bash
py -3.12 -m streamlit run app.py --server.port 8502
```

Open `http://<server-ip>:8502` from another device on the same network; for the whole internet, use a host with a public IP and firewall rules, or use **Streamlit Cloud**.

---

## Fixes applied in the Reflex app

- **Background jobs** (`Run predictions`, minority) now read CSV column state inside `async with self`, so text columns and uploads are not “empty” during inference (a common Reflex bug).
- **`FEEDBACK_MODEL_DIR`** and smarter **relative model path** resolution for servers.
- **`API_URL`** in `rxconfig.py` for public deployments.
