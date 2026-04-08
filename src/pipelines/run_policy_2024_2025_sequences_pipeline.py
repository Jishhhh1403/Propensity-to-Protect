"""Full pipeline for customers who bought any policy in 2024/2025.

Rules implemented:
- Policy events table contains only rows whose ``policy_start_date`` year is 2024 or 2025.
- Daily features table contains **all** customers (so non-buyer negatives remain available), restricted to
  dates in 2024-01-01..2025-12-31.
- ``customer_key`` (INT64) is used across operations and sequence retrieval.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from google.cloud import bigquery

from src.components.config import load_config_from_env
from src.components.feature_engineering import build_feature_frames
from src.components.ingest_from_gcs import load_raw_frames_from_gcs
from src.components.policy_years_filter import (
    eligible_customer_ids_policies_only_in_years,
    policy_rows_for_sequence_export,
    resolve_policy_name,
)
from src.components.preprocess import preprocess_frames
from src.components.publish_to_bigquery import publish_features_to_bigquery
from src.components.sequence_retrieval import fetch_30_day_sequence


def _recreate_bq_tables_from_env() -> bool:
    """Same semantics as :func:`run_all_pipeline._recreate_bq_tables_from_env`."""
    v = os.getenv("PIPELINE_RECREATE_BQ_TABLES", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def _filter_frames_by_customers(
    frames: dict[str, pd.DataFrame], customer_ids: list[str]
) -> dict[str, pd.DataFrame]:
    """Restrict customers / insurance / transactions to ``customer_ids``."""
    customer_id_set = set(customer_ids)
    return {
        "customers": frames["customers"][frames["customers"]["customer_id"].astype(str).isin(customer_id_set)].copy(),
        "insurance": frames["insurance"][frames["insurance"]["customer_id"].astype(str).isin(customer_id_set)].copy(),
        "transactions": frames["transactions"][frames["transactions"]["customer_id"].astype(str).isin(customer_id_set)].copy(),
    }


def _filter_daily_features_calendar_2024_2025(daily: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where date is between 2024-01-01 and 2025-12-31 inclusive."""
    if daily.empty or "date" not in daily.columns:
        return daily
    out = daily.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    start = pd.Timestamp("2024-01-01")
    end = pd.Timestamp("2025-12-31")
    out = out[(out["date"] >= start) & (out["date"] <= end)]
    sort_cols = [c for c in ["customer_key", "customer_id", "date"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def _build_sequence_table_for_policy_events(
    client: bigquery.Client,
    config,
    policy_jobs: pd.DataFrame,
    table_name: str,
) -> None:
    """For each distinct policy purchase row, fetch 30-day pre-purchase features and load one BQ table.

    ``policy_jobs`` must include ``customer_key`` and ``policy_start_date``; ``policy_id`` and
    ``policy_name`` (from ``product_name`` / ``product_type`` on the insurance row) are added
    when present. Duplicate ``(customer_key, policy_start_date)`` rows are collapsed to the
    first row.
    """
    if policy_jobs.empty:
        print("No policy rows in 2024–2025; sequence table not written.")
        return

    jobs = policy_jobs.drop_duplicates(subset=["customer_key", "policy_start_date"], keep="first")
    sequence_frames: list[pd.DataFrame] = []
    skipped_empty = 0

    for _, row in jobs.iterrows():
        # Each policy purchase event becomes one 30-day model window.
        customer_key = int(row["customer_key"])
        ps = pd.Timestamp(row["policy_start_date"])
        as_of_date = ps.date().isoformat()
        seq_df = fetch_30_day_sequence(config, as_of_date=as_of_date, customer_key=customer_key)
        if seq_df.empty:
            skipped_empty += 1
            continue
        seq_df = seq_df.copy()
        seq_df["policy_start_date"] = ps.date()
        if "policy_id" in row.index and pd.notna(row.get("policy_id")):
            seq_df["policy_id"] = str(row["policy_id"])
        # policy_name is needed later for supervised labeling of policy type.
        pn = resolve_policy_name(row)
        seq_df["policy_name"] = pn if pn is not None else pd.NA
        sequence_frames.append(seq_df)

    if not sequence_frames:
        print(
            f"No non-empty 30-day sequences written (attempted {len(jobs)} policies; "
            f"all empty or missing daily_features rows). "
            f"Ensure transaction dates cover the 30 days before each policy start."
        )
        return

    all_sequences = pd.concat(sequence_frames, ignore_index=True)
    all_sequences = all_sequences.sort_values([c for c in ["customer_key", "feature_date"] if c in all_sequences.columns])
    sequence_table_fqn = f"{config.gcp_project_id}.{config.bq_dataset}.{table_name}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(all_sequences, sequence_table_fqn, job_config=job_config).result()
    print(
        f"Saved {len(all_sequences)} sequence rows to {sequence_table_fqn}. "
        f"Skipped empty pulls: {skipped_empty}."
    )


def run(
    sequence_table_name: str = "policy_2024_2025_30_day_sequences",
    years: tuple[int, int] = (2024, 2025),
) -> None:
    """Ingest, publish features, then write 30-day sequences for 2024/2025 policy purchases.

    Args:
        sequence_table_name: BigQuery table name (within ``config.bq_dataset``) for sequence rows.
        years: Inclusive policy years (default 2024 and 2025).
    """
    config = load_config_from_env()
    if not config.gcp_project_id:
        raise ValueError("Set GCP_PROJECT_ID or configure ADC with a default project.")

    print("Loading and preprocessing source data...")
    raw_frames = load_raw_frames_from_gcs(config.gcs_base_uri)
    processed = preprocess_frames(raw_frames)

    eligible = eligible_customer_ids_policies_only_in_years(
        processed["insurance"],
        processed["transactions"],
        years=years,
    )
    if not eligible:
        raise ValueError(
            f"No customers found with policies in {years} and with transactions. "
            "Check insurance dates and policy_start_date coverage."
        )

    print(f"Eligible customers (policies in {years}, with transactions): {len(eligible)}")

    # Build daily_features for all customers so downstream training can sample non-buyers as negatives.
    feature_frames = build_feature_frames(processed)
    feature_frames["daily_features"] = _filter_daily_features_calendar_2024_2025(feature_frames["daily_features"])

    policy_rows = policy_rows_for_sequence_export(
        processed["insurance"],
        eligible,
        years=years,
    )
    feature_frames["policy_events"] = feature_frames["policy_events"][
        feature_frames["policy_events"]["customer_id"].astype(str).isin(set(policy_rows["customer_id"].astype(str)))
    ].copy()
    feature_frames["policy_events"] = feature_frames["policy_events"][
        pd.to_datetime(feature_frames["policy_events"]["purchase_date"], errors="coerce").dt.year.isin(set(years))
    ].copy()
    feature_frames["policy_events"] = feature_frames["policy_events"].sort_values(
        [c for c in ["purchase_date", "customer_key", "customer_id"] if c in feature_frames["policy_events"].columns]
    ).reset_index(drop=True)

    print("Publishing feature tables to BigQuery (daily_features + policy_events will be overwritten)...")
    publish_features_to_bigquery(
        feature_frames,
        config,
        write_disposition="WRITE_TRUNCATE",
        recreate_tables=_recreate_bq_tables_from_env(),
    )

    client = bigquery.Client(project=config.gcp_project_id)
    _build_sequence_table_for_policy_events(
        client=client,
        config=config,
        policy_jobs=policy_rows,
        table_name=sequence_table_name,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full pipeline for 2024–2025-only policy customers and export 30-day sequences."
    )
    parser.add_argument(
        "--sequence-table-name",
        default="policy_2024_2025_30_day_sequences",
        help="BigQuery table name for retrieved 30-day sequences.",
    )
    args = parser.parse_args()
    run(sequence_table_name=args.sequence_table_name)
