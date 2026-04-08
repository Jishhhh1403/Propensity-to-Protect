"""CLI entrypoint: print a single customer's 30-day feature sequence from BigQuery."""

import argparse

from src.components.config import load_config_from_env
from src.components.sequence_retrieval import fetch_30_day_sequence


def run(customer_id: str, as_of_date: str) -> None:
    """Load config, fetch sequence rows, print a preview and row count."""
    config = load_config_from_env()
    if not config.gcp_project_id:
        raise ValueError("Set GCP_PROJECT_ID or configure ADC with a default project.")
    df = fetch_30_day_sequence(config, customer_id=customer_id, as_of_date=as_of_date)
    print(df.head(30))
    print(f"Rows fetched: {len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch one 30-day sequence from BigQuery")
    parser.add_argument("--customer-id", required=True)
    parser.add_argument("--as-of-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    run(customer_id=args.customer_id, as_of_date=args.as_of_date)
