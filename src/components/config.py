"""Load pipeline settings from environment variables and optional project ``.env`` file.

``PipelineConfig`` holds GCP project, GCS paths, BigQuery table names, and Vertex-related
settings used by ingest, publish, and job submission code.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import google.auth


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable bundle of all runtime settings for GCS ingest, BigQuery, and Vertex."""

    gcp_project_id: str
    gcs_bucket: str
    gcs_prefix: str
    bq_dataset: str
    bq_daily_features_table: str
    bq_policy_events_table: str
    bq_policy_sequences_table: str
    bq_propensity_embeddings_table: str
    gcp_region: str
    vertex_pipeline_root: str
    vertex_service_account: str
    vertex_staging_bucket: str
    vertex_image_uri: str
    model_artifacts_gcs_prefix: str

    @property
    def gcs_base_uri(self) -> str:
        """Base ``gs://`` URI for CSV files, with optional object prefix (no trailing slash)."""
        prefix = self.gcs_prefix.strip("/")
        return f"gs://{self.gcs_bucket}/{prefix}" if prefix else f"gs://{self.gcs_bucket}"

    @property
    def daily_features_table_fqn(self) -> str:
        """BigQuery fully qualified name: ``project.dataset.table`` for daily features."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{self.bq_daily_features_table}"

    @property
    def policy_events_table_fqn(self) -> str:
        """BigQuery fully qualified name for the policy events feature table."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{self.bq_policy_events_table}"

    @property
    def policy_sequences_table_fqn(self) -> str:
        """BigQuery fully qualified name for the 30-day sequence table."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{self.bq_policy_sequences_table}"

    @property
    def propensity_embeddings_table_fqn(self) -> str:
        """BigQuery fully qualified name for the GRU propensity embedding output table."""
        return f"{self.gcp_project_id}.{self.bq_dataset}.{self.bq_propensity_embeddings_table}"


def _load_dotenv() -> None:
    """Parse repo-root ``.env`` and set ``os.environ`` keys only if not already defined.

    Lines must look like ``KEY=value``. Quotes around ``value`` are stripped. Comments
    and blank lines are skipped. Does nothing if ``.env`` is missing.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_gcp_project_id(explicit_project_id: str) -> str:
    """Return ``explicit_project_id`` if set; otherwise try Application Default Credentials.

    On Vertex, ADC may resolve to a non-user project, so callers that need a specific
    project should set ``GCP_PROJECT_ID`` (or pass env into the container).
    """
    if explicit_project_id:
        return explicit_project_id
    try:
        _, detected_project = google.auth.default()
    except Exception:
        detected_project = None
    return detected_project or ""


def _resolve_pipeline_root(gcs_bucket: str) -> str:
    """Default GCS URI for Vertex pipeline root artifacts if ``VERTEX_PIPELINE_ROOT`` unset."""
    explicit_root = os.getenv("VERTEX_PIPELINE_ROOT", "").strip()
    if explicit_root:
        return explicit_root
    return f"gs://{gcs_bucket}/vertex/pipeline-root"

def _resolve_model_artifacts_prefix(gcs_bucket: str) -> str:
    """Default GCS prefix for model artifacts if ``MODEL_ARTIFACTS_GCS_PREFIX`` unset."""
    explicit = os.getenv("MODEL_ARTIFACTS_GCS_PREFIX", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return f"gs://{gcs_bucket}/vertex/model-artifacts"


def load_config_from_env() -> PipelineConfig:
    """Load ``.env``, read environment variables, and build a ``PipelineConfig`` instance.

    Uses defaults for bucket/dataset/table names when corresponding env vars are missing.
    """
    _load_dotenv()
    explicit_project_id = os.getenv("GCP_PROJECT_ID", "").strip()
    gcp_project_id = _resolve_gcp_project_id(explicit_project_id)
    gcs_bucket = os.getenv("GCS_BUCKET", "propensitysyntheticdata").strip()
    return PipelineConfig(
        gcp_project_id=gcp_project_id,
        gcs_bucket=gcs_bucket,
        gcs_prefix=os.getenv("GCS_PREFIX", ""),
        bq_dataset=os.getenv("BQ_DATASET", "insurance_features"),
        bq_daily_features_table=os.getenv("BQ_DAILY_FEATURES_TABLE", "daily_features"),
        bq_policy_events_table=os.getenv("BQ_POLICY_EVENTS_TABLE", "policy_events"),
        bq_policy_sequences_table=os.getenv("BQ_POLICY_SEQUENCES_TABLE", "policy_2024_2025_30_day_sequences"),
        bq_propensity_embeddings_table=os.getenv(
            "BQ_PROPENSITY_EMBEDDINGS_TABLE", "policy_propensity_gru_embeddings"
        ),
        gcp_region=os.getenv("GCP_REGION", "us-central1"),
        vertex_pipeline_root=_resolve_pipeline_root(gcs_bucket),
        vertex_service_account=os.getenv("VERTEX_SERVICE_ACCOUNT", "").strip(),
        vertex_staging_bucket=os.getenv("VERTEX_STAGING_BUCKET", "").strip(),
        vertex_image_uri=os.getenv("VERTEX_IMAGE_URI", "").strip(),
        model_artifacts_gcs_prefix=_resolve_model_artifacts_prefix(gcs_bucket),
    )
