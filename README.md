# Feedback Atlas

Student feedback classification and minority pattern detection.

Repository: [github.com/ahmedElmersawy/StudentFeedbackMinority](https://github.com/ahmedElmersawy/StudentFeedbackMinority)

## Models

| Model | Directory | Use case |
|-------|-----------|----------|
| CATME | `catme_feedback_classifier/` | Student → Student (peer/self feedback) |
| Professor | `professor_feedback_classifier/` | Student → Professor (course evaluations) |

Train with:
```bash
sbatch slurm/train_catme.sh
sbatch slurm/train_professor.sh
```

## Running the app

```bash
# Backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend && npm run dev
```

Open: http://localhost:5173 — or see [DEPLOY.md](DEPLOY.md) for Docker Compose.
