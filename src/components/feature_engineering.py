"""Build daily transaction aggregates and policy-event feature tables from preprocessed data."""

from typing import Dict

import numpy as np
import pandas as pd


def _prepare_transaction_daily_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Roll transactions up to one row per (customer_id, calendar date).

    Steps:

    - Floor ``transaction_date`` to midnight → ``date``.
    - Split amounts into inflow (Credit) vs outflow using ``np.where`` (vectorized branch).
    - ``groupby(["customer_id", "date"])`` then sum counts and amounts.
    - Per group, derive category flags and digital payment share via small ``apply`` helpers.

    ``grouped.apply(fn)`` passes each group DataFrame to ``fn``; used where logic is not
    a simple aggregation.
    """
    df = transactions.copy()
    df["date"] = df["transaction_date"].dt.floor("D")
    df["is_inflow"] = df["transaction_type"].isin({"Credit"})
    df["customer_id"] = df["customer_id"].astype(str)
    if "customer_key" in df.columns:
        df["customer_key"] = pd.to_numeric(df["customer_key"], errors="coerce").astype("Int64")
    grouped_cols = ["customer_key", "customer_id", "date"] if "customer_key" in df.columns else ["customer_id", "date"]
    grouped = df.groupby(grouped_cols)

    # Use explicit mask columns to keep compatibility across pandas versions.
    df["amount_inflow"] = np.where(df["is_inflow"], df["amount"], 0.0)
    df["amount_outflow"] = np.where(~df["is_inflow"], df["amount"], 0.0)

    daily_income = grouped["amount_inflow"].sum().rename("daily_income")
    daily_spend = grouped["amount_outflow"].sum().rename("daily_spend")
    tx_count = grouped["transaction_id"].count().rename("tx_count")

    def cat_flags(g: pd.DataFrame) -> pd.Series:
        """Return 0/1 flags if any merchant_category in the day matches health or retail."""
        cats = g["merchant_category"].astype(str).str.lower()
        return pd.Series(
            {
                "cat_flag_health": float(cats.str.contains("health|pharmacy").any()),
                "cat_flag_retail": float(cats.str.contains("grocery|grocer|retail").any()),
            }
        )

    cat_flags_df = grouped.apply(cat_flags)

    def digital_share(g: pd.DataFrame) -> float:
        """Fraction of rows that look digital (method + missing branch/ATM id)."""
        pm = g["payment_method"].astype(str).str.lower()
        digital_methods = ["online transfer", "online/api", "mobile wallet", "bank transfer", "contactless"]
        is_digital = g["branch_atm_id"].isna() & pm.isin(digital_methods)
        return float(is_digital.mean()) if len(g) else 0.0

    channel_digital = grouped.apply(digital_share).rename("channel_digital_pct")
    calendar = grouped["day_of_week"].first().rename("day_of_week")
    is_weekend = (calendar >= 5).astype(float).rename("is_weekend")

    daily = pd.concat([daily_income, daily_spend, tx_count, cat_flags_df, channel_digital, is_weekend], axis=1).reset_index()
    daily[["daily_income", "daily_spend"]] = daily[["daily_income", "daily_spend"]].fillna(0.0)
    return daily


def _add_derived_time_series_features(full_daily: pd.DataFrame) -> pd.DataFrame:
    """Per customer, sort by date and add rolling / cumulative features (balance, z-scores, cyclical day).

    ``groupby(..., group_keys=False).apply(per_customer_features)`` runs the inner function
    on each customer's block of rows. Legacy pandas may leave a mangled index; the code
    below normalizes ``customer_id`` back onto the frame if needed.
    """
    def per_customer_features(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").reset_index(drop=True)
        g["net_flow"] = g["daily_income"] - g["daily_spend"]
        g["daily_balance"] = g["net_flow"].cumsum()
        mean_s = g["daily_spend"].mean()
        g["rel_spend"] = g["daily_spend"] / mean_s if mean_s > 0 else 0.0
        g["bal_trajectory"] = g["daily_balance"].diff(7).fillna(0.0) / 7
        rolling_tx = g["tx_count"].rolling(window=7, min_periods=1).mean()
        g["velocity_ratio"] = np.where(rolling_tx > 0, g["tx_count"] / rolling_tx, 0.0)
        std_s = g["daily_spend"].std()
        g["spend_z_score"] = (g["daily_spend"] - mean_s) / std_s if std_s > 0 else 0.0
        dom = g["date"].dt.day.astype(float)
        g["day_sin"] = np.sin(2 * np.pi * dom / 31.0)
        g["day_cos"] = np.cos(2 * np.pi * dom / 31.0)
        return g

    full_daily = full_daily.copy()
    full_daily["customer_id"] = full_daily["customer_id"].astype(str)
    if "customer_key" in full_daily.columns:
        full_daily["customer_key"] = pd.to_numeric(full_daily["customer_key"], errors="coerce").astype("Int64")
    group_col = "customer_key" if "customer_key" in full_daily.columns else "customer_id"
    out = full_daily.groupby(group_col, group_keys=False).apply(per_customer_features)
    if "customer_id" not in out.columns:
        out = out.reset_index()
    if "customer_id" not in out.columns and "level_0" in out.columns:
        out = out.rename(columns={"level_0": "customer_id"})
    return out


def build_daily_feature_table(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Build the daily_features table: dense calendar per customer, derived signals, join static attrs.

    For each customer, ``pd.date_range(min, max, freq="D")`` plus ``reindex`` inserts rows
    for days with no transactions (filled with 0). Then merges selected columns from
    ``customers`` (age, income_band, etc.).
    """
    daily_agg = _prepare_transaction_daily_aggregates(transactions)
    expanded = []
    group_col = "customer_key" if "customer_key" in daily_agg.columns else "customer_id"
    for key, g in daily_agg.groupby(group_col):
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        add_cols = {"customer_id": str(g["customer_id"].iloc[0])}
        if group_col == "customer_key":
            add_cols["customer_key"] = int(key) if pd.notna(key) else pd.NA
        expanded.append(g.set_index("date").reindex(full_idx).rename_axis("date").reset_index().assign(**add_cols))
    if not expanded:
        return pd.DataFrame()
    full_daily = pd.concat(expanded, ignore_index=True).fillna(0)
    full_daily["customer_id"] = full_daily["customer_id"].astype(str)
    if "customer_key" in full_daily.columns:
        full_daily["customer_key"] = pd.to_numeric(full_daily["customer_key"], errors="coerce").astype("Int64")
    daily = _add_derived_time_series_features(full_daily)
    if "customer_id" not in daily.columns:
        daily = daily.reset_index()
        if "customer_id" not in daily.columns and "level_0" in daily.columns:
            daily = daily.rename(columns={"level_0": "customer_id"})
    daily["customer_id"] = daily["customer_id"].astype(str)
    static_cols = ["customer_key", "customer_id", "age", "income_band", "credit_score", "current_income", "city"]
    customers_static = customers[[c for c in static_cols if c in customers.columns]]
    if "customer_key" in customers_static.columns:
        customers_static = customers_static.copy()
        customers_static["customer_key"] = pd.to_numeric(customers_static["customer_key"], errors="coerce").astype("Int64")
    if "customer_id" in customers_static.columns:
        customers_static = customers_static.copy()
        customers_static["customer_id"] = customers_static["customer_id"].astype(str)
    join_col = "customer_key" if "customer_key" in daily.columns and "customer_key" in customers_static.columns else "customer_id"
    daily = daily.merge(customers_static, on=join_col, how="left", suffixes=("", "_static"))
    if "customer_id_static" in daily.columns:
        daily["customer_id"] = daily["customer_id_static"].fillna(daily["customer_id"])
        daily = daily.drop(columns=["customer_id_static"])
    sort_cols = [c for c in ["customer_key", "customer_id", "date"] if c in daily.columns]
    return daily.sort_values(sort_cols).reset_index(drop=True) if sort_cols else daily


def build_policy_events_table(insurance: pd.DataFrame) -> pd.DataFrame:
    """Select policy/insurance columns and map ``launch_date`` / ``policy_start_date`` to event/purchase dates."""
    df = insurance.copy()
    df["event_date"] = pd.to_datetime(df["launch_date"], errors="coerce") if "launch_date" in df.columns else pd.NaT
    df["purchase_date"] = (
        pd.to_datetime(df["policy_start_date"], errors="coerce") if "policy_start_date" in df.columns else pd.NaT
    )
    cols = [
        "customer_key",
        "policy_id",
        "customer_id",
        "event_date",
        "purchase_date",
        "product_id",
        "product_name",
        "product_type",
        "premium_amount",
        "coverage_amount",
        "policy_status",
        "risk_category",
    ]
    out = df[[c for c in cols if c in df.columns]].copy()
    if "customer_key" in out.columns:
        out["customer_key"] = pd.to_numeric(out["customer_key"], errors="coerce").astype("Int64")
    sort_cols = [c for c in ["purchase_date", "customer_key", "customer_id", "event_date"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


def build_feature_frames(processed_frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Return ``daily_features`` and ``policy_events`` DataFrames from preprocessed inputs."""
    return {
        "daily_features": build_daily_feature_table(processed_frames["customers"], processed_frames["transactions"]),
        "policy_events": build_policy_events_table(processed_frames["insurance"]),
    }
