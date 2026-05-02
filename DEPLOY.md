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

1. Push the repo to GitHub (include `final_feedback_classifier/` or download weights in a startup script).
2. On [share.streamlit.io](https://share.streamlit.io), create an app with main file `app.py`.
3. Set Python **3.12** and `requirements.txt`.

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
