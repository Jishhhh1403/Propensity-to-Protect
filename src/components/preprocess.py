"""Clean and type-normalize raw customer, insurance, and transaction DataFrames."""

from typing import Dict

import numpy as np
import pandas as pd
import re


def _add_customer_key(df: pd.DataFrame) -> pd.DataFrame:
    """Add integer ``customer_key`` derived from ``customer_id`` digits.

    Uses pandas nullable integer dtype ``Int64`` so missing keys remain null-compatible.
    """
    if "customer_id" not in df.columns:
        return df
    out = df.copy()
    s = out["customer_id"].astype("string")
    digits = s.str.extract(r"(\d+)$", expand=False)
    out["customer_key"] = pd.to_numeric(digits, errors="coerce").astype("Int64")
    return out


def _ensure_customer_id_str(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``customer_id`` to a canonical digits-only string.

    Source files sometimes use different formats for the same customer:
    - digit-only IDs: ``605``
    - prefixed/zero-padded IDs: ``CUST_000605``

    For joins (and 30-day sequence retrieval), the exact string must match across tables.
    We normalize both formats to digits-only with leading zeros removed:
    - ``CUST_000605`` → ``605``
    - ``000605`` → ``605``
    - ``605`` → ``605``

    Returns ``df`` unchanged if ``customer_id`` column is missing.
    """
    if "customer_id" not in df.columns:
        return df
    out = df.copy()
    cust_prefix_re = re.compile(r"^cust_(\d+)$", flags=re.IGNORECASE)
    digits_only_re = re.compile(r"^\d+$")

    def canonicalize(v: object) -> object:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return pd.NA
        s = str(v).strip()
        if s in ("", "nan", "None", "NaN"):
            return pd.NA
        m = cust_prefix_re.match(s)
        if m:
            digits = m.group(1)
        elif digits_only_re.match(s):
            digits = s
        else:
            return s
        return digits.lstrip("0") or "0"

    out["customer_id"] = out["customer_id"].map(canonicalize)
    return out


def preprocess_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Strip object columns, parse dates and numeric fields, fix ``income_band``, sort by id.

    ``format="mixed"`` in ``to_datetime`` allows multiple date string formats in one column.
    """
    df = customers.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    if "account_open_date" in df.columns:
        df["account_open_date"] = pd.to_datetime(df["account_open_date"], errors="coerce", format="mixed")
    for col in ["age", "credit_score", "current_income"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "income_band" in df.columns:
        df["income_band"] = df["income_band"].replace({np.nan: "Unknown", "": "Unknown"})
    df = _ensure_customer_id_str(df)
    df = _add_customer_key(df)
    return df.sort_values(["customer_key", "customer_id"]).reset_index(drop=True) if "customer_id" in df.columns else df


def preprocess_insurance(insurance: pd.DataFrame) -> pd.DataFrame:
    """Clean insurance rows: strings, dates, numerics, boolean-like ``medical_requirement``, sort."""
    df = insurance.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    for date_col in ["launch_date", "policy_start_date"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    for col in ["coverage_amount", "premium_amount", "policy_term_years", "beneficiary_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "medical_requirement" in df.columns:
        df["medical_requirement"] = (
            df["medical_requirement"].astype(str).str.strip().str.lower().replace({"true": True, "false": False})
        )
    sort_cols = [c for c in ["customer_id", "policy_id", "policy_start_date"] if c in df.columns]
    df = _ensure_customer_id_str(df)
    df = _add_customer_key(df)
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def preprocess_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Clean transactions: strings, ``transaction_date``, numerics, empty ``branch_atm_id`` → NaN."""
    df = transactions.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    if "transaction_date" in df.columns:
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce", format="mixed")
    for col in ["amount", "month", "day_of_week"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "branch_atm_id" in df.columns:
        df["branch_atm_id"] = df["branch_atm_id"].replace({"": np.nan})
    sort_cols = [c for c in ["customer_id", "transaction_date"] if c in df.columns]
    df = _ensure_customer_id_str(df)
    df = _add_customer_key(df)
    return df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df


def preprocess_frames(raw_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Run the three preprocessors on ``raw_frames`` from :func:`load_raw_frames_from_gcs`."""
    customers = preprocess_customers(raw_frames["customers"])
    insurance = preprocess_insurance(raw_frames["insurance"])
    transactions = preprocess_transactions(raw_frames["transactions"])
    return {"customers": customers, "insurance": insurance, "transactions": transactions}
