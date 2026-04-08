"""Submit a Vertex AI Custom Job that runs a pipeline module inside your Artifact Registry image.

By default the job runs :mod:`run_policy_2024_2025_sequences_pipeline` (must match ``Dockerfile`` ``ENTRYPOINT``).
Use ``--pipeline run_all`` for the legacy sampled test pipeline.
"""

import argparse

from google.cloud import aiplatform

from src.components.config import PipelineConfig, load_config_from_env

# ``python -m <module>`` passed to the container; overrides image ENTRYPOINT when set.
_PIPELINE_MODULES = {
    "policy_2024_2025": "src.pipelines.run_policy_2024_2025_sequences_pipeline",
    "run_all": "src.pipelines.run_all_pipeline",
    "build_features": "src.pipelines.build_features_pipeline",
    "gru_propensity": "src.pipelines.run_gru_propensity_pipeline",
}


def _vertex_worker_env(config: PipelineConfig) -> list[dict[str, str]]:
    """Build ``container_spec.env`` so the worker uses the intended GCP project and data settings.

    Vertex's default credentials sometimes associate a different *project* with ADC; setting
    ``GCP_PROJECT_ID`` and ``GOOGLE_CLOUD_PROJECT`` forces BigQuery/GCS clients to bill and
    resolve resources in the right project.

    Omits any pair whose value is empty: the Vertex API rejects env vars with blank values.

    Returns:
        List of ``{"name": "...", "value": "..."}`` dicts for the Custom Job API.
    """
    pairs = [
        ("GCP_PROJECT_ID", config.gcp_project_id),
        ("GOOGLE_CLOUD_PROJECT", config.gcp_project_id),
        ("GCS_BUCKET", config.gcs_bucket),
        ("GCS_PREFIX", config.gcs_prefix),
        ("BQ_DATASET", config.bq_dataset),
        ("BQ_DAILY_FEATURES_TABLE", config.bq_daily_features_table),
        ("BQ_POLICY_EVENTS_TABLE", config.bq_policy_events_table),
        ("BQ_POLICY_SEQUENCES_TABLE", config.bq_policy_sequences_table),
        ("BQ_PROPENSITY_EMBEDDINGS_TABLE", config.bq_propensity_embeddings_table),
        ("MODEL_ARTIFACTS_GCS_PREFIX", config.model_artifacts_gcs_prefix),
        ("GCP_REGION", config.gcp_region),
        # Avoid bigquery.tables.delete; WRITE_TRUNCATE is enough for managed runs.
        ("PIPELINE_RECREATE_BQ_TABLES", "false"),
    ]
    # Vertex rejects env entries with an empty value (proto "required field is not set").
    out: list[dict[str, str]] = []
    for k, v in pairs:
        s = v.strip() if isinstance(v, str) else str(v).strip()
        if s:
            # Vertex API rejects blank env values, so we filter them here.
            out.append({"name": k, "value": s})
    return out


def _vertex_staging_bucket_uri(config: PipelineConfig) -> str:
    """Return the GCS URI Vertex uses as ``staging_bucket`` for Custom Job metadata.

    Must be in the **same regional location** as ``config.gcp_region`` (e.g. us-central1).
    Multi-region ``US`` buckets used for raw data are invalid for that constraint, so when
    ``VERTEX_STAGING_BUCKET`` is unset we default to ``gs://<project-id>-vertex-ai-staging``.
    """
    raw = (config.vertex_staging_bucket or "").strip()
    if raw:
        return raw if raw.startswith("gs://") else f"gs://{raw}"
    safe = config.gcp_project_id.strip().lower().replace("_", "-")
    return f"gs://{safe}-vertex-ai-staging"


def run(
    display_name: str,
    worker_pool_machine_type: str = "n1-standard-4",
    pipeline: str = "policy_2024_2025",
) -> None:
    """Initialize Vertex SDK, create a Custom Job, and wait until it finishes (``sync=True``).

    Args:
        display_name: Shown in the Vertex / console UI.
        worker_pool_machine_type: GCE machine type for the single worker (e.g. ``n1-standard-4``).
        pipeline: Key into ``_PIPELINE_MODULES`` (default: 2024–2025 cohort + sequences).
    """
    config = load_config_from_env()
    image_uri = config.vertex_image_uri
    if not config.gcp_project_id:
        raise ValueError("Set GCP_PROJECT_ID in .env or environment, or configure ADC with a default project.")
    if not image_uri:
        raise ValueError("Set VERTEX_IMAGE_URI to an Artifact Registry image before submitting a Vertex job.")
    if pipeline not in _PIPELINE_MODULES:
        raise ValueError(f"Unknown pipeline {pipeline!r}. Choose from {sorted(_PIPELINE_MODULES)}.")
    # This module path is executed inside the container as: python -m <module>.
    pipeline_module = _PIPELINE_MODULES[pipeline]

    staging_bucket = _vertex_staging_bucket_uri(config)
    aiplatform.init(
        project=config.gcp_project_id,
        location=config.gcp_region,
        staging_bucket=staging_bucket,
    )

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[
            {
                "machine_spec": {"machine_type": worker_pool_machine_type},
                "replica_count": 1,
                "container_spec": {
                    "image_uri": image_uri,
                    "command": ["python", "-m", pipeline_module],
                    "env": _vertex_worker_env(config),
                },
            }
        ],
    )

    run_kwargs: dict = {"sync": True}
    if config.vertex_service_account:
        run_kwargs["service_account"] = config.vertex_service_account

    job.run(**run_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit a pipeline module to Vertex AI Custom Job (default: 2024–2025 sequences pipeline)."
    )
    parser.add_argument(
        "--display-name",
        default="cloud-propensity-policy-2024-2025",
        help="Display name for the Vertex AI Custom Job run.",
    )
    parser.add_argument(
        "--machine-type",
        default="n1-standard-4",
        help="Vertex AI worker machine type (example: n1-standard-4).",
    )
    parser.add_argument(
        "--pipeline",
        choices=sorted(_PIPELINE_MODULES.keys()),
        default="policy_2024_2025",
        help="Which pipeline to run inside the container (must exist in the image).",
    )
    args = parser.parse_args()
    run(display_name=args.display_name, worker_pool_machine_type=args.machine_type, pipeline=args.pipeline)
