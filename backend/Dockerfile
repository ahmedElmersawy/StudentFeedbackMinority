# ── Build stage: install heavy deps separately so layer is cached ──────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# ── Runtime stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY backend/ /app/backend/

# Models: baked into the image as a fallback, then symlinked from the paths
# the app actually reads (/app/<name>_feedback_classifier). A single volume
# mounted at /app/models (Railway allows one volume/service) shadows this
# baked-in copy with real weights uploaded post-deploy.
COPY catme_feedback_classifier/ /app/models/catme_feedback_classifier/
COPY professor_feedback_classifier/ /app/models/professor_feedback_classifier/
RUN ln -s /app/models/catme_feedback_classifier /app/catme_feedback_classifier \
 && ln -s /app/models/professor_feedback_classifier /app/professor_feedback_classifier

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]
