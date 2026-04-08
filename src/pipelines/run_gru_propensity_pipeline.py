"""Train a GRU policy-type model from the 30-day sequence table and export embeddings to BigQuery.

Runs entirely on BigQuery + GCS + Vertex (when submitted as a Custom Job).

Inputs (env/config):
- ``BQ_POLICY_SEQUENCES_TABLE``: sequence table name (default set in config)
- ``MODEL_ARTIFACTS_GCS_PREFIX``: where to save SavedModel + artifacts JSON

Outputs (env/config):
- ``BQ_PROPENSITY_EMBEDDINGS_TABLE``: output table name (default set in config)
"""

from __future__ import annotations

import argparse
import os

from src.components.config import load_config_from_env
from src.components.gru_propensity import GRUTrainingParams, run_gru_training_and_export


def run(
    *,
    epochs: int = 30,
    embedding_dim: int = 64,
    gru_units: int = 64,
    dense_units: int = 64,
    batch_size: int = 16,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    val_fraction: float = 0.2,
    negative_per_positive: float = 1.0,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    config = load_config_from_env()
    if not config.gcp_project_id:
        raise ValueError("Set GCP_PROJECT_ID or configure ADC with a default project.")

    params = GRUTrainingParams(
        embedding_dim=embedding_dim,
        gru_units=gru_units,
        dense_units=dense_units,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
    )

    result = run_gru_training_and_export(
        bq_project=config.gcp_project_id,
        sequences_table_fqn=config.policy_sequences_table_fqn,
        daily_features_table_fqn=config.daily_features_table_fqn,
        policy_events_table_fqn=config.policy_events_table_fqn,
        output_table_fqn=config.propensity_embeddings_table_fqn,
        model_artifacts_gcs_prefix=config.model_artifacts_gcs_prefix,
        params=params,
        negative_per_positive=negative_per_positive,
        write_disposition=write_disposition,
    )
    print(f"GRU training/export complete. Output table: {result['output_table']}. Run: {result['run_id']}.")
    print(f"Saved model to: {result['model_gcs_prefix']}")
    print(f"Saved artifacts JSON to: {result['artifacts_json']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train GRU on 30-day sequences and export embeddings to BigQuery.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--gru-units", type=int, default=64)
    p.add_argument("--dense-units", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument(
        "--negative-per-positive",
        type=float,
        default=float(os.getenv("NEGATIVE_PER_POSITIVE", "1") or "1"),
        help="How many non-buyer negative sequences to add per positive policy event (default from env NEGATIVE_PER_POSITIVE).",
    )
    p.add_argument("--write-disposition", default="WRITE_TRUNCATE", choices=["WRITE_TRUNCATE", "WRITE_APPEND"])
    args = p.parse_args()
    run(
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        gru_units=args.gru_units,
        dense_units=args.dense_units,
        batch_size=args.batch_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        negative_per_positive=args.negative_per_positive,
        write_disposition=args.write_disposition,
    )

