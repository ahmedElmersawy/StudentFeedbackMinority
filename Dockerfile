# Reflex app: frontend ~3000, backend ~8000 (set API_URL to http://HOST:8000 before build/run).
FROM python:3.12-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY rxconfig.py feedback_core.py ./
COPY feedback_web ./feedback_web/

# Optional: bake a small classifier into the image (uncomment if repo includes it).
# COPY final_feedback_classifier ./final_feedback_classifier/

ARG API_URL=
ENV API_URL=${API_URL}

RUN reflex compile

EXPOSE 3000 8000

# Backend must listen on all interfaces so the public API_URL is reachable.
CMD ["reflex", "run", "--env", "prod", "--backend-host", "0.0.0.0"]
