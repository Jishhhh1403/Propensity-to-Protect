"""Load raw CSV tables from Google Cloud Storage into pandas DataFrames."""

from typing import Dict

import pandas as pd


def load_raw_frames_from_gcs(gcs_base_uri: str) -> Dict[str, pd.DataFrame]:
    """Read the three source CSVs from GCS under ``gcs_base_uri``.

    Expects these paths (same folder as the URI base):

    - ``customers_data.csv``
    - ``insurance_data.csv``
    - ``transactions_data.csv``

    ``pandas.read_csv`` accepts ``gs://`` URLs when ``gcsfs`` (via fsspec) is installed.
    ID columns are forced to pandas ``string`` dtype so values like ``CUST_001`` are
    not parsed as numbers and joins stay consistent with BigQuery.

    Args:
        gcs_base_uri: e.g. ``gs://my-bucket`` or ``gs://my-bucket/prefix`` (no trailing slash required).

    Returns:
        Dict with keys ``customers``, ``insurance``, ``transactions``.
    """
    base = gcs_base_uri.rstrip("/")
    # Force ID-like columns to strings to prevent pandas from inferring numeric IDs
    # (which can break joins and BigQuery sequence retrieval).
    customers = pd.read_csv(
        f"{base}/customers_data.csv",
        dtype={"customer_id": "string"},
        keep_default_na=True,
    )
    insurance = pd.read_csv(
        f"{base}/insurance_data.csv",
        dtype={"customer_id": "string", "policy_id": "string"},
        keep_default_na=True,
    )
    transactions = pd.read_csv(
        f"{base}/transactions_data.csv",
        dtype={"customer_id": "string", "transaction_id": "string"},
        keep_default_na=True,
    )
    for df in (customers, insurance, transactions):
        df.columns = [c.strip() for c in df.columns]
    return {"customers": customers, "insurance": insurance, "transactions": transactions}
