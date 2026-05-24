FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source
COPY backend/ /app/backend/
COPY feedback_core.py /app/feedback_core.py

# Copy model if present (or mount at runtime)
COPY final_feedback_classifier/ /app/final_feedback_classifier/

ENV FEEDBACK_MODEL_DIR=/app/final_feedback_classifier
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
