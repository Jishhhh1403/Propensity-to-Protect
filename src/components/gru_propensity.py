"""Train a GRU policy-type classifier on 30-day sequences and export embeddings to BigQuery.

This module expects a BigQuery *sequence* table in long format (one row per day) that includes
the columns produced by :func:`src.components.sequence_retrieval.fetch_30_day_sequence` plus
policy metadata columns added by the policy sequences pipeline:

- ``customer_key`` (INT64)
- ``customer_id`` (STRING)
- ``policy_start_date`` (DATE)
- ``policy_name`` (STRING)
- ``feature_date`` (DATE)
- feature columns (mostly numeric)

The training objective is multi-class (softmax) over policy types (``policy_name``).
The exported "behavior/propensity vector" is the GRU final hidden state for each policy event.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from google.cloud import bigquery

_NO_PURCHASE_POLICY_NAME = "__NO_PURCHASE__"


@dataclass(frozen=True)
class GRUTrainingParams:
    embedding_dim: int = 64
    gru_units: int = 64
    dense_units: int = 64
    dropout: float = 0.1
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 7


def _utc_now_ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _policy_name_to_class_id(policy_names: Iterable[str]) -> dict[str, int]:
    # Deterministic class index mapping is critical so probabilities remain interpretable across runs.
    uniq = sorted({str(p).strip() for p in policy_names if str(p).strip()})
    return {name: i for i, name in enumerate(uniq)}


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Pick model feature columns from a raw sequence frame.

    We intentionally avoid categorical/string columns for v1. If you want to include them later,
    add explicit encoding (one-hot / embeddings) and persist the mapping.
    """
    exclude = {
        "customer_key",
        "customer_id",
        "as_of_date",
        "policy_start_date",
        "policy_id",
        "policy_name",
        "feature_date",
        "day_offset",
    }
    candidates = [c for c in df.columns if c not in exclude]
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    # Stable ordering for reproducibility
    return sorted(numeric)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized x, mean, std. x is (N, T, F)."""
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std > 0, std, 1.0)
    xz = (x - mean) / std
    return xz, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std


def load_sequences_from_bigquery(
    client: bigquery.Client,
    table_fqn: str,
) -> pd.DataFrame:
    # The model trains only on rows that have an explicit policy label.
    query = f"""
    SELECT *
    FROM `{table_fqn}`
    WHERE policy_name IS NOT NULL
    ORDER BY customer_key, policy_start_date, feature_date
    """
    rows = client.query(query).result()
    df = pd.DataFrame([dict(r.items()) for r in rows])
    if df.empty:
        return df
    # Normalize dtypes
    for c in ("policy_start_date", "feature_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    if "customer_key" in df.columns:
        df["customer_key"] = pd.to_numeric(df["customer_key"], errors="coerce").astype("Int64")
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)
    df["policy_name"] = df["policy_name"].astype(str)
    return df


def _fetch_30_day_sequence_from_daily_features(
    *,
    client: bigquery.Client,
    daily_features_table_fqn: str,
    customer_key: int,
    customer_id: str | None,
    as_of_date: str,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Fetch 30-day window from daily_features for negative examples."""
    select_features = ",\n      ".join(feature_cols)
    query = f"""
    SELECT
      customer_key,
      customer_id,
      @as_of_date AS as_of_date,
      date AS feature_date,
      DATE_DIFF(date, @as_of_date, DAY) AS day_offset,
      {select_features}
    FROM `{daily_features_table_fqn}`
    WHERE customer_key = @customer_key
      AND date BETWEEN DATE_SUB(@as_of_date, INTERVAL 30 DAY)
                   AND DATE_SUB(@as_of_date, INTERVAL 1 DAY)
    ORDER BY date
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("customer_key", "INT64", int(customer_key)),
            bigquery.ScalarQueryParameter("as_of_date", "DATE", as_of_date),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    df = pd.DataFrame([dict(r.items()) for r in rows])
    if df.empty:
        return df
    if customer_id is not None and "customer_id" in df.columns:
        # Prefer the provided stable id if the table holds mixed formats.
        df["customer_id"] = str(customer_id)
    return df


def load_negative_sequences_from_bigquery(
    *,
    client: bigquery.Client,
    daily_features_table_fqn: str,
    policy_events_table_fqn: str,
    feature_cols: list[str],
    n_sequences: int,
) -> pd.DataFrame:
    """Sample `n_sequences` non-buyer customers and pull one 30-day window each.

    A "non-buyer" is a customer_key that appears in daily_features but never appears in policy_events.
    We use each customer's max available daily_features date as the window end (as_of_date = max_date + 1 day).
    """
    if n_sequences <= 0:
        return pd.DataFrame()

    query = f"""
    WITH buyers AS (
      SELECT DISTINCT customer_key
      FROM `{policy_events_table_fqn}`
      WHERE customer_key IS NOT NULL AND purchase_date IS NOT NULL
    ),
    nonbuyers AS (
      SELECT
        customer_key,
        ANY_VALUE(customer_id) AS customer_id,
        MAX(date) AS max_date
      FROM `{daily_features_table_fqn}`
      WHERE customer_key IS NOT NULL
        AND customer_key NOT IN (SELECT customer_key FROM buyers)
      GROUP BY customer_key
    )
    SELECT customer_key, customer_id, max_date
    FROM nonbuyers
    ORDER BY RAND()
    LIMIT @n
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("n", "INT64", int(n_sequences))]
    )
    rows = list(client.query(query, job_config=job_config).result())
    if not rows:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for r in rows:
        customer_key = int(r["customer_key"])
        customer_id = str(r.get("customer_id")) if r.get("customer_id") is not None else None
        max_date = pd.to_datetime(r["max_date"], errors="coerce")
        if pd.isna(max_date):
            continue
        as_of_date = (max_date + pd.Timedelta(days=1)).date().isoformat()
        seq = _fetch_30_day_sequence_from_daily_features(
            client=client,
            daily_features_table_fqn=daily_features_table_fqn,
            customer_key=customer_key,
            customer_id=customer_id,
            as_of_date=as_of_date,
            feature_cols=feature_cols,
        )
        if seq.empty or len(seq) != 30:
            # Keep the training tensor strict: every event must have exactly 30 daily rows.
            continue
        seq = seq.copy()
        seq["policy_start_date"] = pd.to_datetime(as_of_date).date()
        seq["policy_id"] = pd.NA
        seq["policy_name"] = _NO_PURCHASE_POLICY_NAME
        frames.append(seq)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # Normalize date dtypes like positives
    for c in ("policy_start_date", "feature_date"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce").dt.date
    return out


def build_training_tensors(
    sequences_long: pd.DataFrame,
    feature_cols: list[str] | None = None,
    expected_days: int = 30,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """Convert long sequences to (X, y, meta_df, feature_cols).

    - X: (N, 30, F)
    - y: (N,) int class ids
    - meta_df: one row per sequence/event with join keys and label strings
    """
    if sequences_long.empty:
        return np.zeros((0, expected_days, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64), pd.DataFrame(), []

    df = sequences_long.copy()
    if feature_cols is None:
        feature_cols = _numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found in sequence table to train on.")

    # Ensure numeric + fill missing
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)

    group_cols = ["customer_key", "policy_start_date", "policy_name"]
    if "policy_id" in df.columns:
        group_cols.append("policy_id")

    policy_to_id = _policy_name_to_class_id(df["policy_name"].tolist())

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    meta_rows: list[dict] = []

    for keys, g in df.groupby(group_cols, dropna=False):
        g = g.sort_values("feature_date").reset_index(drop=True)
        if len(g) != expected_days:
            # Skip incomplete windows (should be rare if retrieval query is consistent).
            continue
        x = g[feature_cols].to_numpy(dtype=np.float32)
        if x.shape != (expected_days, len(feature_cols)):
            continue

        # Unpack keys depending on whether policy_id is present
        if "policy_id" in df.columns:
            customer_key, policy_start_date, policy_name, policy_id = keys
        else:
            customer_key, policy_start_date, policy_name = keys
            policy_id = None

        X_list.append(x)
        y_list.append(policy_to_id[str(policy_name)])
        meta_rows.append(
            {
                "customer_key": int(customer_key) if pd.notna(customer_key) else None,
                "customer_id": str(g["customer_id"].iloc[0]) if "customer_id" in g.columns else None,
                "policy_start_date": policy_start_date,
                "policy_name": str(policy_name),
                "policy_id": str(policy_id) if policy_id is not None and str(policy_id) != "nan" else None,
            }
        )

    if not X_list:
        raise ValueError("No complete 30-day sequences found to train on (all windows incomplete).")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df, feature_cols


def train_gru_softmax(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    params: GRUTrainingParams,
) -> tuple[object, object, np.ndarray, np.ndarray]:
    """Train and return (keras_model, embedder_model, mean, std)."""
    # Lazy import so feature-only pipelines don't require TF unless you run this module.
    import tensorflow as tf

    tf.random.set_seed(params.seed)
    np.random.seed(params.seed)

    Xz, mean, std = _standardize_fit(X)

    # Split
    n = Xz.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(params.seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * params.val_fraction))) if n >= 5 else 1
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        # Tiny-data fallback to avoid crashing on very small cohorts.
        train_idx = val_idx
        val_idx = idx[:0]

    X_train, y_train = Xz[train_idx], y[train_idx]
    X_val, y_val = Xz[val_idx], y[val_idx]

    # Model
    inp = tf.keras.Input(shape=(X.shape[1], X.shape[2]), name="sequence")
    x = tf.keras.layers.GRU(
        params.gru_units,
        dropout=params.dropout,
        return_sequences=False,
        name="gru",
    )(inp)
    emb = tf.keras.layers.Dense(params.embedding_dim, activation="tanh", name="embedding")(x)
    h = tf.keras.layers.Dense(params.dense_units, activation="relu", name="dense")(emb)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="policy_probs")(h)

    model = tf.keras.Model(inputs=inp, outputs=out, name="policy_gru_softmax")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "batch_size": params.batch_size,
        "epochs": params.epochs,
        "verbose": 2,
        "callbacks": callbacks,
    }
    if len(val_idx) > 0:
        fit_kwargs["validation_data"] = (X_val, y_val)
    model.fit(**fit_kwargs)

    embedder = tf.keras.Model(inputs=inp, outputs=model.get_layer("embedding").output, name="policy_gru_embedder")
    return model, embedder, mean.astype(np.float32), std.astype(np.float32)


def score_embeddings(
    embedder_model: object,
    classifier_model: object,
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    Xz = _standardize_apply(X, mean, std)
    emb = tf.convert_to_tensor(Xz, dtype=tf.float32)
    embeddings = embedder_model.predict(emb, verbose=0)
    probs = classifier_model.predict(emb, verbose=0)
    return np.asarray(embeddings), np.asarray(probs)


def save_artifacts_to_gcs(
    *,
    classifier_model: object,
    feature_cols: list[str],
    policy_name_to_id: dict[str, int],
    mean: np.ndarray,
    std: np.ndarray,
    gcs_prefix: str,
    run_id: str,
) -> dict[str, str]:
    """Save model + preprocessing artifacts to GCS via gcsfs.

    Returns dict with artifact URIs.
    """
    import gcsfs

    base = f"{gcs_prefix.rstrip('/')}/policy_gru/{run_id}"
    fs = gcsfs.GCSFileSystem()

    # Save as native Keras format to avoid TF SavedModel export issues
    # with untracked resources in certain runtime combinations.
    local_model_path = os.path.join("/tmp", f"policy_gru_{run_id}.keras")
    classifier_model.save(local_model_path)
    model_uri = f"{base}/model.keras"
    fs.put(local_model_path, model_uri)

    meta = {
        "run_id": run_id,
        "created_at": _utc_now_ts(),
        # Persist preprocessing metadata so future scoring uses identical feature order/scaling.
        "feature_cols": feature_cols,
        "policy_name_to_id": policy_name_to_id,
        "standardize_mean": mean.tolist(),
        "standardize_std": std.tolist(),
    }
    meta_uri = f"{base}/artifacts.json"
    with fs.open(meta_uri, "w") as f:
        json.dump(meta, f, indent=2)

    return {"model_gcs_prefix": model_uri, "artifacts_json": meta_uri}


def write_embeddings_to_bigquery(
    client: bigquery.Client,
    table_fqn: str,
    out_df: pd.DataFrame,
    write_disposition: str = "WRITE_TRUNCATE",
) -> None:
    if out_df.empty:
        raise ValueError("No embedding rows to write.")
    # Ensure table exists with a stable schema before load.
    dataset_fqn = ".".join(table_fqn.split(".")[:2])
    client.create_dataset(bigquery.Dataset(dataset_fqn), exists_ok=True)

    desired_schema = [
        bigquery.SchemaField("customer_key", "INT64"),
        bigquery.SchemaField("customer_id", "STRING"),
        bigquery.SchemaField("policy_start_date", "DATE"),
        bigquery.SchemaField("policy_name", "STRING"),
        bigquery.SchemaField("policy_id", "STRING"),
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("model_version", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
        bigquery.SchemaField("class_names_json", "STRING"),
        bigquery.SchemaField("embedding_vector", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("probability_vector", "FLOAT64", mode="REPEATED"),
    ]

    try:
        table = client.get_table(table_fqn)
        existing = {f.name for f in table.schema}
        missing = [f for f in desired_schema if f.name not in existing]
        if missing:
            table.schema = list(table.schema) + missing
            client.update_table(table, ["schema"])
    except Exception:
        # First run: create a partitioned/clustering-friendly table for downstream analytics.
        table = bigquery.Table(table_fqn, schema=desired_schema)
        table.time_partitioning = bigquery.TimePartitioning(field="policy_start_date")
        table.clustering_fields = ["customer_id", "policy_name"]
        client.create_table(table, exists_ok=True)

    aligned = out_df.copy()
    # Add any missing columns expected by schema
    for f in desired_schema:
        if f.name not in aligned.columns:
            aligned[f.name] = pd.NA
    aligned = aligned[[f.name for f in desired_schema]]

    job_config = bigquery.LoadJobConfig(schema=desired_schema, write_disposition=write_disposition)
    client.load_table_from_dataframe(aligned, table_fqn, job_config=job_config).result()


def build_embeddings_output_frame(
    meta_df: pd.DataFrame,
    embeddings: np.ndarray,
    probs: np.ndarray,
    policy_id_to_name: dict[int, str],
    run_id: str,
    model_version: str,
) -> pd.DataFrame:
    if len(meta_df) != embeddings.shape[0] or len(meta_df) != probs.shape[0]:
        raise ValueError("Meta rows and scored tensors are misaligned.")
    created_at = _utc_now_ts()
    out = meta_df.copy()
    out["run_id"] = run_id
    out["model_version"] = model_version
    out["created_at"] = pd.to_datetime(created_at)
    # Store class order explicitly; probability_vector indexes map to this JSON array.
    out["class_names_json"] = json.dumps([policy_id_to_name[i] for i in range(probs.shape[1])])
    out["embedding_vector"] = [embeddings[i, :].astype(float).tolist() for i in range(embeddings.shape[0])]
    out["probability_vector"] = [probs[i, :].astype(float).tolist() for i in range(probs.shape[0])]
    return out


def run_gru_training_and_export(
    *,
    bq_project: str,
    sequences_table_fqn: str,
    daily_features_table_fqn: str,
    policy_events_table_fqn: str,
    output_table_fqn: str,
    model_artifacts_gcs_prefix: str,
    params: GRUTrainingParams | None = None,
    negative_per_positive: float = 1.0,
    write_disposition: str = "WRITE_TRUNCATE",
) -> dict[str, str]:
    """End-to-end: load sequences -> build tensors -> train -> score -> write embeddings table."""
    params = params or GRUTrainingParams()
    run_id = uuid.uuid4().hex[:12]
    client = bigquery.Client(project=bq_project)

    sequences = load_sequences_from_bigquery(client, sequences_table_fqn)
    if sequences.empty:
        raise ValueError(f"No rows found in sequences table {sequences_table_fqn}.")

    # Determine numeric feature columns from positives, then optionally add negative sequences.
    _, _, _, feature_cols = build_training_tensors(sequences, feature_cols=None)
    try:
        positive_events = sequences.drop_duplicates(subset=["customer_key", "policy_start_date", "policy_name"])
        pos_count = len(positive_events)
    except Exception:
        pos_count = max(1, int(len(sequences) / 30))
    neg_target = int(max(0, round(pos_count * float(negative_per_positive))))
    negatives = load_negative_sequences_from_bigquery(
        client=client,
        daily_features_table_fqn=daily_features_table_fqn,
        policy_events_table_fqn=policy_events_table_fqn,
        feature_cols=feature_cols,
        n_sequences=neg_target,
    )
    if not negatives.empty:
        # Train on combined positives + sampled non-buyers.
        sequences = pd.concat([sequences, negatives], ignore_index=True)

    # Build tensors on combined set
    X, y, meta_df, feature_cols = build_training_tensors(sequences, feature_cols=feature_cols)
    policy_name_to_id = _policy_name_to_class_id(meta_df["policy_name"].tolist())
    id_to_name = {v: k for k, v in policy_name_to_id.items()}

    model, embedder, mean, std = train_gru_softmax(X, y, n_classes=len(policy_name_to_id), params=params)
    embeddings, probs = score_embeddings(embedder, model, X, mean, std)

    # Save artifacts (optional but strongly recommended on Vertex)
    artifacts = save_artifacts_to_gcs(
        classifier_model=model,
        feature_cols=feature_cols,
        policy_name_to_id=policy_name_to_id,
        mean=mean,
        std=std,
        gcs_prefix=model_artifacts_gcs_prefix,
        run_id=run_id,
    )

    out_df = build_embeddings_output_frame(
        meta_df=meta_df,
        embeddings=embeddings,
        probs=probs,
        policy_id_to_name=id_to_name,
        run_id=run_id,
        model_version=f"policy_gru_softmax_v1:{run_id}",
    )
    write_embeddings_to_bigquery(client, output_table_fqn, out_df, write_disposition=write_disposition)
    return {"run_id": run_id, **artifacts, "output_table": output_table_fqn}

