"""Sampled end-to-end test: GCS → features → BigQuery, then 30-day sequences for buyer customers."""

import argparse
import os
from typing import Iterable

import pandas as pd
from google.cloud import bigquery

from src.components.config import load_config_from_env
from src.components.feature_engineering import build_feature_frames
from src.components.ingest_from_gcs import load_raw_frames_from_gcs
from src.components.preprocess import preprocess_frames
from src.components.publish_to_bigquery import publish_features_to_bigquery
from src.components.policy_years_filter import resolve_policy_name
from src.components.sequence_retrieval import fetch_30_day_sequence


def _recreate_bq_tables_from_env() -> bool:
    """Whether to drop BigQuery tables before load (needs ``bigquery.tables.delete``).

    Env ``PIPELINE_RECREATE_BQ_TABLES``: truthy (``1``, ``true``, …) or falsy (``0``, ``false``, …).
    If **unset**, defaults to ``True`` for backward-compatible local runs. Vertex submit sets ``false``.
    """
    v = os.getenv("PIPELINE_RECREATE_BQ_TABLES", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return True


def _sample_test_customers(
    customers: pd.DataFrame,
    insurance: pd.DataFrame,
    transactions: pd.DataFrame,
    buyers_count: int = 3,
    nonbuyers_count: int = 2,
) -> tuple[list[str], list[str]]:
    """Pick ``buyers_count`` customers with a policy start date and txns, plus ``nonbuyers_count`` without.

    *Buyers* = distinct ``customer_id`` in insurance with non-null ``policy_start_date`` and
    present in transactions. *Non-buyers* = other customers (not in buyers) who still have
    transactions, for a balanced mini sample.

    Raises:
        ValueError: If not enough distinct customers match the criteria.
    """
    tx_customer_ids = set(transactions["customer_id"].dropna().astype(str).unique())
    insurance_df = insurance.copy()
    insurance_df["customer_id"] = insurance_df["customer_id"].astype(str)
    insurance_df["policy_start_date"] = pd.to_datetime(insurance_df.get("policy_start_date"), errors="coerce")

    buyers = sorted(
        cid
        for cid in insurance_df.loc[insurance_df["policy_start_date"].notna(), "customer_id"].unique()
        if cid in tx_customer_ids
    )[:buyers_count]

    customer_ids = customers["customer_id"].dropna().astype(str).unique()
    nonbuyers = [cid for cid in sorted(customer_ids) if cid not in set(buyers) and cid in tx_customer_ids][:nonbuyers_count]

    if len(buyers) < buyers_count:
        raise ValueError(f"Not enough buyer customers with transactions. Needed {buyers_count}, found {len(buyers)}.")
    if len(nonbuyers) < nonbuyers_count:
        raise ValueError(
            f"Not enough non-buyer customers with transactions. Needed {nonbuyers_count}, found {len(nonbuyers)}."
        )

    return buyers, nonbuyers


def _filter_frames_by_customers(
    raw_frames: dict[str, pd.DataFrame], customer_ids: Iterable[str]
) -> dict[str, pd.DataFrame]:
    """Restrict customers / insurance / transactions DataFrames to the given ``customer_ids``."""
    customer_id_set = set(customer_ids)
    return {
        "customers": raw_frames["customers"][raw_frames["customers"]["customer_id"].astype(str).isin(customer_id_set)].copy(),
        "insurance": raw_frames["insurance"][raw_frames["insurance"]["customer_id"].astype(str).isin(customer_id_set)].copy(),
        "transactions": raw_frames["transactions"][
            raw_frames["transactions"]["customer_id"].astype(str).isin(customer_id_set)
        ].copy(),
    }


def _build_sequence_table_for_buyers(
    client: bigquery.Client,
    config,
    insurance: pd.DataFrame,
    buyers: list[str],
    table_name: str,
) -> None:
    """For each buyer, fetch 30-day pre-purchase features and load one concatenated BigQuery table.

    Uses the **earliest** ``policy_start_date`` per buyer as ``as_of_date`` for
    :func:`fetch_30_day_sequence`. Table is written with ``WRITE_TRUNCATE``.
    """
    insurance_df = insurance.copy()
    insurance_df["customer_id"] = insurance_df["customer_id"].astype(str)
    insurance_df["policy_start_date"] = pd.to_datetime(insurance_df.get("policy_start_date"), errors="coerce")

    sequence_frames = []
    for customer_id in buyers:
        purchase_dates = (
            insurance_df.loc[
                (insurance_df["customer_id"] == customer_id) & insurance_df["policy_start_date"].notna(),
                "policy_start_date",
            ]
            .sort_values()
            .unique()
        )
        if len(purchase_dates) == 0:
            continue
        as_of_date = pd.Timestamp(purchase_dates[0]).date().isoformat()
        first_ts = pd.Timestamp(purchase_dates[0]).normalize()
        match = insurance_df[
            (insurance_df["customer_id"] == customer_id)
            & (pd.to_datetime(insurance_df["policy_start_date"], errors="coerce").dt.normalize() == first_ts)
        ]
        pn = resolve_policy_name(match.iloc[0]) if not match.empty else None
        seq_df = fetch_30_day_sequence(config, customer_id=customer_id, as_of_date=as_of_date)
        if not seq_df.empty:
            seq_df = seq_df.copy()
            seq_df["policy_name"] = pn if pn is not None else pd.NA
            sequence_frames.append(seq_df)

    if not sequence_frames:
        print("No 30-day sequences found for sampled buyers; sequence table not written.")
        return

    all_sequences = pd.concat(sequence_frames, ignore_index=True)
    sequence_table_fqn = f"{config.gcp_project_id}.{config.bq_dataset}.{table_name}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    client.load_table_from_dataframe(all_sequences, sequence_table_fqn, job_config=job_config).result()
    print(f"Saved {len(all_sequences)} sequence rows to {sequence_table_fqn}.")


def run(
    buyers_count: int = 3,
    nonbuyers_count: int = 2,
    sequence_table_name: str = "policy_purchase_30_day_sequences",
) -> None:
    """Run sampled pipeline: ingest, subset customers, features, publish, then sequence export for buyers.

    Args:
        buyers_count: Number of policy buyers to include in the sample.
        nonbuyers_count: Number of non-buyers with transactions to include.
        sequence_table_name: BigQuery table name (within ``config.bq_dataset``) for sequence rows.
    """
    config = load_config_from_env()
    if not config.gcp_project_id:
        raise ValueError("Set GCP_PROJECT_ID or configure ADC with a default project.")

    print("Loading and preprocessing source data...")
    raw_frames = load_raw_frames_from_gcs(config.gcs_base_uri)
    processed = preprocess_frames(raw_frames)

    buyers, nonbuyers = _sample_test_customers(
        customers=processed["customers"],
        insurance=processed["insurance"],
        transactions=processed["transactions"],
        buyers_count=buyers_count,
        nonbuyers_count=nonbuyers_count,
    )
    selected_customer_ids = buyers + nonbuyers
    print(f"Selected customers: buyers={buyers}, nonbuyers={nonbuyers}")

    sampled_frames = _filter_frames_by_customers(processed, selected_customer_ids)
    feature_frames = build_feature_frames(sampled_frames)

    print("Publishing sampled feature store tables to BigQuery...")
    publish_features_to_bigquery(
        feature_frames,
        config,
        write_disposition="WRITE_TRUNCATE",
        recreate_tables=_recreate_bq_tables_from_env(),
    )

    client = bigquery.Client(project=config.gcp_project_id)
    _build_sequence_table_for_buyers(
        client=client,
        config=config,
        insurance=sampled_frames["insurance"],
        buyers=buyers,
        table_name=sequence_table_name,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run fast test pipeline for sampled customers (default: 3 buyers + 2 non-buyers)."
    )
    parser.add_argument("--buyers-count", type=int, default=3, help="Number of policy buyers to sample.")
    parser.add_argument("--nonbuyers-count", type=int, default=2, help="Number of non-buyers to sample.")
    parser.add_argument(
        "--sequence-table-name",
        default="policy_purchase_30_day_sequences",
        help="BigQuery table name for retrieved 30-day sequences.",
    )
    args = parser.parse_args()

    run(
        buyers_count=args.buyers_count,
        nonbuyers_count=args.nonbuyers_count,
        sequence_table_name=args.sequence_table_name,
    )
