"""Create BigQuery dataset/tables if needed and load feature DataFrames with partitioning/clustering."""

from typing import Dict

import pandas as pd
from google.cloud import bigquery

from src.components.config import PipelineConfig


def _ensure_dataset(client: bigquery.Client, dataset_fqn: str) -> None:
    """Create the dataset ``dataset_fqn`` (``project.dataset``) in location US if missing.

    ``exists_ok=True`` avoids errors when the dataset already exists.
    """
    dataset = bigquery.Dataset(dataset_fqn)
    dataset.location = "US"
    client.create_dataset(dataset, exists_ok=True)


def _daily_features_schema() -> list[bigquery.SchemaField]:
    """Schema for the ``daily_features`` table (types must match load job expectations)."""
    return [
        bigquery.SchemaField("customer_key", "INT64"),
        bigquery.SchemaField("customer_id", "STRING"),
        bigquery.SchemaField("date", "DATE"),
        bigquery.SchemaField("daily_income", "FLOAT"),
        bigquery.SchemaField("daily_spend", "FLOAT"),
        bigquery.SchemaField("tx_count", "INT64"),
        bigquery.SchemaField("cat_flag_health", "FLOAT"),
        bigquery.SchemaField("cat_flag_retail", "FLOAT"),
        bigquery.SchemaField("channel_digital_pct", "FLOAT"),
        bigquery.SchemaField("is_weekend", "FLOAT"),
        bigquery.SchemaField("net_flow", "FLOAT"),
        bigquery.SchemaField("daily_balance", "FLOAT"),
        bigquery.SchemaField("rel_spend", "FLOAT"),
        bigquery.SchemaField("bal_trajectory", "FLOAT"),
        bigquery.SchemaField("velocity_ratio", "FLOAT"),
        bigquery.SchemaField("spend_z_score", "FLOAT"),
        bigquery.SchemaField("day_sin", "FLOAT"),
        bigquery.SchemaField("day_cos", "FLOAT"),
        bigquery.SchemaField("age", "FLOAT"),
        bigquery.SchemaField("income_band", "STRING"),
        bigquery.SchemaField("credit_score", "FLOAT"),
        bigquery.SchemaField("current_income", "FLOAT"),
        bigquery.SchemaField("city", "STRING"),
    ]


def _policy_events_schema() -> list[bigquery.SchemaField]:
    """Schema for the ``policy_events`` table."""
    return [
        bigquery.SchemaField("customer_key", "INT64"),
        bigquery.SchemaField("policy_id", "STRING"),
        bigquery.SchemaField("customer_id", "STRING"),
        bigquery.SchemaField("event_date", "DATE"),
        bigquery.SchemaField("purchase_date", "DATE"),
        bigquery.SchemaField("product_id", "STRING"),
        bigquery.SchemaField("product_name", "STRING"),
        bigquery.SchemaField("product_type", "STRING"),
        bigquery.SchemaField("premium_amount", "FLOAT"),
        bigquery.SchemaField("coverage_amount", "FLOAT"),
        bigquery.SchemaField("policy_status", "STRING"),
        bigquery.SchemaField("risk_category", "STRING"),
    ]


def _align_df_to_schema(df: pd.DataFrame, schema: list[bigquery.SchemaField]) -> pd.DataFrame:
    """Add missing columns as ``pd.NA`` and reorder columns to match ``schema`` field order."""
    cols = [field.name for field in schema]
    aligned = df.copy()
    for col in cols:
        if col not in aligned.columns:
            # Missing columns are added as nullable values so load jobs do not fail on schema mismatch.
            aligned[col] = pd.NA
    return aligned[cols]


def _ensure_table_and_optional_columns(
    client: bigquery.Client,
    table_fqn: str,
    desired_schema: list[bigquery.SchemaField],
    partition_field: str,
    clustering_fields: list[str],
) -> bigquery.Table:
    """Create table if missing; if present, add any missing nullable columns."""
    try:
        table = client.get_table(table_fqn)
        existing = {f.name for f in table.schema}
        missing = [f for f in desired_schema if f.name not in existing]
        if missing:
            table.schema = list(table.schema) + missing
            table = client.update_table(table, ["schema"])
        return table
    except Exception:
        table = bigquery.Table(table_fqn)
        table.schema = desired_schema
        table.time_partitioning = bigquery.TimePartitioning(field=partition_field)
        table.clustering_fields = clustering_fields
        return client.create_table(table, exists_ok=True)


def publish_features_to_bigquery(
    feature_frames: Dict[str, pd.DataFrame],
    config: PipelineConfig,
    write_disposition: str = "WRITE_TRUNCATE",
    recreate_tables: bool = False,
) -> None:
    """Load ``daily_features`` and ``policy_events`` DataFrames into BigQuery.

    Ensures dataset exists, creates tables with time partitioning and clustering if needed,
    then runs ``load_table_from_dataframe`` for each table.

    Args:
        feature_frames: Must contain keys ``daily_features`` and ``policy_events``.
        config: Project and table FQNs from :class:`~src.components.config.PipelineConfig`.
        write_disposition: e.g. ``WRITE_TRUNCATE`` replaces table contents (see BigQuery docs).
        recreate_tables: If True, **deletes** existing tables first (requires
            ``bigquery.tables.delete`` permission); use False on restricted service accounts.
    """
    client = bigquery.Client(project=config.gcp_project_id)
    _ensure_dataset(client, f"{config.gcp_project_id}.{config.bq_dataset}")

    daily_schema = _daily_features_schema()
    policy_schema = _policy_events_schema()

    daily_df = _align_df_to_schema(feature_frames["daily_features"], daily_schema)
    policy_df = _align_df_to_schema(feature_frames["policy_events"], policy_schema)

    if "customer_key" in daily_df.columns:
        daily_df["customer_key"] = pd.to_numeric(daily_df["customer_key"], errors="coerce").astype("Int64")
    if "customer_key" in policy_df.columns:
        policy_df["customer_key"] = pd.to_numeric(policy_df["customer_key"], errors="coerce").astype("Int64")

    if recreate_tables:
        # Optional hard reset path for local runs with full table-recreate permissions.
        client.delete_table(config.daily_features_table_fqn, not_found_ok=True)
        client.delete_table(config.policy_events_table_fqn, not_found_ok=True)

    _ensure_table_and_optional_columns(
        client=client,
        table_fqn=config.daily_features_table_fqn,
        desired_schema=daily_schema,
        partition_field="date",
        clustering_fields=["customer_id"],
    )
    _ensure_table_and_optional_columns(
        client=client,
        table_fqn=config.policy_events_table_fqn,
        desired_schema=policy_schema,
        partition_field="event_date",
        clustering_fields=["customer_id"],
    )

    client.load_table_from_dataframe(
        daily_df,
        config.daily_features_table_fqn,
        job_config=bigquery.LoadJobConfig(
            schema=daily_schema,
            write_disposition=write_disposition,
        ),
    ).result()

    client.load_table_from_dataframe(
        policy_df,
        config.policy_events_table_fqn,
        job_config=bigquery.LoadJobConfig(
            schema=policy_schema,
            write_disposition=write_disposition,
        ),
    ).result()
