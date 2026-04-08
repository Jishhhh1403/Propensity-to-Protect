# Cloud Propensity Pipeline

End-to-end Cloud pipeline for insurance propensity modeling with Google Cloud.

This project:
- Ingests raw CSV-like source data from GCS
- Cleans and engineers daily customer features in Python
- Publishes feature tables to BigQuery
- Builds 30-day pre-purchase sequences
- Trains a GRU model and exports embeddings/predictions to BigQuery
- Supports both local execution and Vertex AI Custom Job execution

## High-level flow

1. Load config from environment (`.env`).
2. Read source frames from GCS (`customers`, `insurance`, `transactions`).
3. Preprocess and feature engineer in Python.
4. Write feature tables to BigQuery (`daily_features`, `policy_events`).
5. Retrieve 30-day windows from BigQuery for each policy event.
6. Train GRU and write embeddings + probabilities back to BigQuery.

## Repository structure

- `src/components/`
  - `config.py`: configuration loader and environment parsing
  - `ingest_from_gcs.py`: source ingestion from GCS
  - `preprocess.py`: data cleanup and normalization
  - `feature_engineering.py`: daily feature generation
  - `publish_to_bigquery.py`: BigQuery table publishing
  - `sequence_retrieval.py`: 30-day sequence query logic
  - `gru_propensity.py`: GRU train/infer/export pipeline logic
- `src/pipelines/`
  - `build_features_pipeline.py`: feature-only pipeline
  - `get_sequence_pipeline.py`: fetch one customer sequence
  - `run_all_pipeline.py`: sampled full pipeline test run
  - `run_policy_2024_2025_sequences_pipeline.py`: full 2024/2025 policy sequence generation
  - `run_gru_propensity_pipeline.py`: GRU training + embedding export
  - `run_on_vertex_custom_job.py`: submit selected pipeline to Vertex AI
- `sql/`: SQL artifacts (if needed for manual experimentation)
- `docs/`: non-technical and workflow documentation
- `Dockerfile`: runtime image definition for Vertex custom job

## Prerequisites

- Python 3.10+ (recommended)
- Google Cloud project with:
  - BigQuery enabled
  - Vertex AI enabled (if using custom jobs)
  - GCS bucket with source data
- Service account key file (or ADC) with required permissions

## Quick start for teammates

1. Clone repo:
   - `git clone https://github.com/Jishhhh1403/Propensity-to-Protect.git`
   - `cd Propensity-to-Protect`
2. Create virtual environment:
   - Windows PowerShell:
     - `python -m venv .venv`
     - `.venv\Scripts\Activate.ps1`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Configure environment:
   - Copy `.env.example` to `.env`
   - Fill required values (project, dataset, tables, GCS inputs, credentials path)
5. Authenticate:
   - `gcloud auth activate-service-account --key-file "<path-to-key.json>"`
   - or configure ADC and default project:
     - `gcloud auth application-default login`
     - `gcloud config set project <PROJECT_ID>`

## Environment variables

Use `.env.example` as the canonical template. Important variables include:
- `GCP_PROJECT_ID`
- `GOOGLE_APPLICATION_CREDENTIALS`
- `GCS_BUCKET`, `GCS_PREFIX`
- `BQ_DATASET`
- `BQ_DAILY_FEATURES_TABLE`, `BQ_POLICY_EVENTS_TABLE`
- `BQ_POLICY_SEQUENCES_TABLE`
- `BQ_PROPENSITY_EMBEDDINGS_TABLE`
- `MODEL_ARTIFACTS_GCS_PREFIX`
- `VERTEX_IMAGE_URI` (for Vertex run)

## How to run

### 1) Build and publish features only

`python -m src.pipelines.build_features_pipeline`

### 2) Fetch one 30-day sequence (debug/validation)

`python -m src.pipelines.get_sequence_pipeline --customer-id CUST_000001 --as-of-date 2025-01-15`

### 3) Run full 2024/2025 sequence generation

`python -m src.pipelines.run_policy_2024_2025_sequences_pipeline`

This pipeline:
- filters policy events for 2024-2025
- keeps daily features for full population in 2024-2025 window
- writes `daily_features` + `policy_events` to BigQuery
- exports 30-day sequence rows with `policy_name`

### 4) Train GRU and export propensity embeddings

`python -m src.pipelines.run_gru_propensity_pipeline`

### 5) Sampled full local run (quick smoke)

`python -m src.pipelines.run_all_pipeline`

## Vertex AI custom job run

1. Build and push image:
   - `gcloud auth configure-docker us-central1-docker.pkg.dev`
   - `docker build -t us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO>/cloud-propensity:latest .`
   - `docker push us-central1-docker.pkg.dev/<PROJECT_ID>/<REPO>/cloud-propensity:latest`
2. Set `VERTEX_IMAGE_URI` in `.env`.
3. Submit full run:
   - `python -m src.pipelines.run_on_vertex_custom_job --display-name cloud-propensity-run-all`
4. Submit GRU-only run:
   - `python -m src.pipelines.run_on_vertex_custom_job --display-name cloud-propensity-gru-propensity --pipeline gru_propensity`

## Typical execution order

For a fresh environment:
1. `build_features_pipeline`
2. `run_policy_2024_2025_sequences_pipeline`
3. `run_gru_propensity_pipeline`

For quick debugging:
1. `run_all_pipeline`
2. `get_sequence_pipeline`

## Notes

- `.env` is intentionally ignored by git.
- `.env.example` should be kept up-to-date and committed for teammates.
- If no `GCP_PROJECT_ID` is explicitly set, code can use ADC default project where supported.
