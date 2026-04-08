"""Select customers/policies whose ``policy_start_date`` is in given calendar years (e.g. 2024–2025)."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def resolve_policy_name(row: pd.Series) -> str | None:
    """Return a display label for the policy line used in sequence / training metadata.

    Prefers ``product_name`` from insurance source data, then ``product_type``. Used to populate
    ``policy_name`` on 30-day sequence rows aligned with each policy purchase.
    """
    for col in ("product_name", "product_type"):
        if col not in row.index:
            continue
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if not s or s.lower() in ("nan", "none", "<na>"):
            continue
        return s
    return None


def eligible_customer_ids_policies_only_in_years(
    insurance: pd.DataFrame,
    transactions: pd.DataFrame,
    years: Iterable[int] = (2024, 2025),
) -> list[str]:
    """Return ``customer_id`` values that have at least one policy_start_date in ``years``.

    Rules:

    - Consider only rows with non-null ``policy_start_date``.
    - Include customer if **any** ``policy_start_date`` year is in ``years``.
    - The customer must appear in ``transactions`` (so we can build daily features).

    Args:
        insurance: Preprocessed insurance frame with ``customer_id``, ``policy_start_date``.
        transactions: Used to require at least one transaction row per customer.
        years: Allowed calendar years (default 2024 and 2025).

    Returns:
        Sorted list of ``customer_id`` strings.
    """
    allowed = set(int(y) for y in years)
    ins = insurance.copy()
    ins["customer_id"] = ins["customer_id"].astype(str)
    ins["policy_start_date"] = pd.to_datetime(ins.get("policy_start_date"), errors="coerce")
    dated = ins.dropna(subset=["policy_start_date"])
    if dated.empty:
        return []

    tx_ids = set(transactions["customer_id"].dropna().astype(str).unique())

    in_years = dated[dated["policy_start_date"].dt.year.isin(allowed)]
    eligible_ids = set(in_years["customer_id"].astype(str).unique()) & tx_ids
    return sorted(eligible_ids)


def policy_rows_for_sequence_export(
    insurance: pd.DataFrame,
    eligible_customer_ids: list[str],
    years: Iterable[int] = (2024, 2025),
) -> pd.DataFrame:
    """Insurance rows for eligible customers with ``policy_start_date`` in ``years``.

    One row per underlying policy line; caller may dedupe on ``(customer_key, policy_start_date)``
    if needed. Includes ``policy_id``, ``product_name``, and ``product_type`` when present (used
    to populate ``policy_name`` on sequence rows).
    """
    allowed = set(int(y) for y in years)
    eligible_set = set(eligible_customer_ids)
    ins = insurance.copy()
    ins["customer_id"] = ins["customer_id"].astype(str)
    ins["policy_start_date"] = pd.to_datetime(ins.get("policy_start_date"), errors="coerce")
    ins = ins[ins["customer_id"].isin(eligible_set)].dropna(subset=["policy_start_date"])
    ins = ins[ins["policy_start_date"].dt.year.isin(allowed)]
    if "customer_key" in ins.columns:
        ins["customer_key"] = pd.to_numeric(ins["customer_key"], errors="coerce").astype("Int64")
    return ins.sort_values([c for c in ["policy_start_date", "customer_key", "customer_id", "policy_id"] if c in ins.columns]).reset_index(
        drop=True
    )
