"""End-to-end pipeline: GCS → preprocess → features → BigQuery (all rows in source files)."""

from src.components.config import load_config_from_env
from src.components.feature_engineering import build_feature_frames
from src.components.ingest_from_gcs import load_raw_frames_from_gcs
from src.components.preprocess import preprocess_frames
from src.components.publish_to_bigquery import publish_features_to_bigquery


def run() -> None:
    """Load config, ingest raw CSVs from GCS, build features, publish both tables to BigQuery."""
    config = load_config_from_env()
    if not config.gcp_project_id:
        raise ValueError("Set GCP_PROJECT_ID or configure ADC with a default project.")
    raw_frames = load_raw_frames_from_gcs(config.gcs_base_uri)
    processed_frames = preprocess_frames(raw_frames)
    feature_frames = build_feature_frames(processed_frames)
    publish_features_to_bigquery(feature_frames, config)


if __name__ == "__main__":
    run()
