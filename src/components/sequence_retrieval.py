"""Query BigQuery for a 30-day daily feature window before an as-of date."""

import pandas as pd
from google.cloud import bigquery

from src.components.config import PipelineConfig


def fetch_30_day_sequence(
    config: PipelineConfig,
    as_of_date: str,
    customer_id: str | None = None,
    customer_key: int | None = None,
) -> pd.DataFrame:
    """Return daily feature rows for one customer for the 30 days **before** ``as_of_date``.

    The SQL uses **parameterized** queries (``@customer_id``, ``@as_of_date``) so values
    are sent separately from the query text—this avoids SQL injection and handles types
    (``DATE``) correctly.

    The window is ``DATE_SUB(as_of_date, 30 days)`` through ``DATE_SUB(as_of_date, 1 day)``
    (exclusive of the as-of day itself).

    Args:
        config: Used for ``daily_features_table_fqn`` and BigQuery ``project``.
        customer_id: Optional string customer identifier (legacy path).
        customer_key: Optional integer customer key (preferred).
        as_of_date: ``YYYY-MM-DD`` string interpreted as BigQuery DATE.

    Returns:
        DataFrame with one row per calendar day in the window (may be empty).
    """
    if customer_key is None and customer_id is None:
        raise ValueError("Provide either customer_key or customer_id.")

    client = bigquery.Client(project=config.gcp_project_id)
    # Keep this SELECT schema stable; downstream GRU code expects these feature columns.
    query = f"""
    SELECT
      customer_key,
      customer_id,
      @as_of_date AS as_of_date,
      date AS feature_date,
      DATE_DIFF(date, @as_of_date, DAY) AS day_offset,
      daily_income,
      daily_spend,
      tx_count,
      cat_flag_health,
      cat_flag_retail,
      channel_digital_pct,
      is_weekend,
      net_flow,
      daily_balance,
      rel_spend,
      bal_trajectory,
      velocity_ratio,
      spend_z_score,
      day_sin,
      day_cos,
      age,
      income_band,
      credit_score,
      current_income,
      city
    FROM `{config.daily_features_table_fqn}`
    WHERE (
        (@customer_key IS NOT NULL AND customer_key = @customer_key)
        OR
        (@customer_key IS NULL AND customer_id = @customer_id)
    )
      AND date BETWEEN DATE_SUB(@as_of_date, INTERVAL 30 DAY)
                   AND DATE_SUB(@as_of_date, INTERVAL 1 DAY)
    ORDER BY date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
            bigquery.ScalarQueryParameter("customer_key", "INT64", customer_key),
            bigquery.ScalarQueryParameter("as_of_date", "DATE", as_of_date),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return pd.DataFrame([dict(row.items()) for row in rows])
