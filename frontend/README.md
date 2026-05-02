# Feedback Atlas — React dashboard

## Run locally

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Build

```bash
npm run build
npm run preview   # serve production build
```

## Connect the real model

1. Implement a backend route (e.g. FastAPI) that runs your Python `feedback_core` pipeline.
2. In `src/api/mockDetect.ts`, replace `runMockMinorityDetection` with a `fetch("/api/detect-minority", …)` call.
3. Configure a Vite proxy in `vite.config.ts` for local dev, or set `VITE_API_BASE` and use it in the fetch URL.

The UI already sends: parsed rows, selected feedback columns, optional rating & date columns — mirror that in your API contract.
