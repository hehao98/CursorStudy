#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from google.cloud import bigquery

REPOS_CSV = Path(__file__).parent.parent / "data" / "repos.csv"
REPO_EVENTS_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "repo_events.csv"
START_DATE = None
END_DATE = "250419"
REPO_QUERY = """
SELECT type, created_at, repo.name as repo, actor.login as actor
FROM `githubarchive.day.20*`
WHERE repo.name IN UNNEST(@repos)
ORDER BY repo, created_at
"""


def load_repos_with_cursor_adoption() -> List[str]:
    """Load repos with Cursor adoption dates from CSV."""
    df = pd.read_csv(REPOS_CSV)
    cursor_repos = df[df["repo_cursor_adoption"].notna()]
    repos = cursor_repos["repo_name"].tolist()
    logging.info(f"Loaded {len(repos)} repos with Cursor adoption dates")
    return repos


def format_bytes(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["bytes", "KB", "MB", "GB"]:
        if bytes_val < 1024 or unit == "GB":
            return (
                f"{bytes_val:.2f} {unit}" if unit != "bytes" else f"{bytes_val} {unit}"
            )
        bytes_val /= 1024
    return f"{bytes_val:.2f} GB"


def estimate_query_cost(repos: List[str]) -> Tuple[float, float]:
    """Estimate BigQuery cost with a dry run."""
    if not repos:
        return 0, 0

    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("repos", "STRING", repos)],
        dry_run=True,
        use_query_cache=False,
    )

    query_job = client.query(REPO_QUERY, job_config=job_config)
    bytes_processed = query_job.total_bytes_processed
    terabytes = bytes_processed / (1024**4)
    cost = terabytes * 5.0

    return bytes_processed, cost


def fetch_events_from_bigquery(repos: List[str], output_path: str) -> None:
    """Fetch repo events from BigQuery and save to CSV."""
    if not repos:
        logging.warning("No repos with Cursor adoption dates found")
        return

    logging.info(f"Preparing to fetch events for {len(repos)} repos")

    # Estimate query cost
    bytes_processed, cost = estimate_query_cost(repos)
    bytes_str = format_bytes(bytes_processed)
    logging.info(f"Estimated query size: {bytes_str}")
    logging.info(f"Estimated cost: ${cost:.4f} USD")

    # Ask for confirmation if cost is significant
    if cost > 0.01:
        prompt = f"Proceed with query (cost ~${cost:.4f})? (y/n): "
        if input(prompt).lower() != "y":
            logging.info("Query cancelled")
            return

    # Run the actual query
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("repos", "STRING", repos)]
    )

    logging.info("Executing query...")
    results = client.query(REPO_QUERY, job_config=job_config).result()
    df_results = results.to_dataframe()

    if df_results.empty:
        logging.warning("No events found")
        return

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df_results)} events to {output_path}")

    # Log basic stats
    logging.info("Events by type:")
    for event_type, count in df_results["type"].value_counts().items():
        logging.info(f"  {event_type}: {count}")

    logging.info("Top 10 repos by event count:")
    for repo, count in df_results["repo"].value_counts().head(10).items():
        logging.info(f"  {repo}: {count}")


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    repos = load_repos_with_cursor_adoption()
    fetch_events_from_bigquery(repos, REPO_EVENTS_OUTPUT_FILE)


if __name__ == "__main__":
    main()
