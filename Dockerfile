FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY sql /app/sql

# Default container command for Vertex AI: full 2024–2025 policy cohort + 30-day sequences.
# Override at run time: docker run ... python -m src.pipelines.run_all_pipeline
ENTRYPOINT ["python", "-m", "src.pipelines.run_policy_2024_2025_sequences_pipeline"]
