#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from google.cloud import bigquery

REPOS_CSV = Path(__file__).parent.parent / "data" / "repos.csv"
REPO_EVENTS_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "repo_events.csv"
TIME_KEY = None
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
    return bytes_processed, bytes_processed / (1024**4) * 5.0


def fetch_events_from_bigquery(repos: List[str], output_path: str) -> None:
    """Fetch repo events from BigQuery and save to CSV."""
    if not repos:
        logging.warning("No repos with Cursor adoption dates found")
        return

    logging.info(f"Preparing to fetch events for {len(repos)} repos")
    bytes_processed, cost = estimate_query_cost(repos)
    logging.info(f"Estimated query size: {format_bytes(bytes_processed)}")
    logging.info(f"Estimated cost: ${cost:.4f} USD")

    if (
        cost > 0.01
        and input(f"Proceed with query (cost ~${cost:.4f})? (y/n): ").lower() != "y"
    ):
        logging.info("Query cancelled")
        return

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df_results)} events to {output_path}")

    logging.info("Events by type:")
    for event_type, count in df_results["type"].value_counts().items():
        logging.info(f"  {event_type}: {count}")

    logging.info("Top 10 repos by event count:")
    for repo, count in df_results["repo"].value_counts().head(10).items():
        logging.info(f"  {repo}: {count}")


def compute_event_metrics(events_df: pd.DataFrame, time_format: str) -> pd.DataFrame:
    """Compute event metrics from GitHub events data."""
    events_df["time_period"] = events_df["created_at"].dt.strftime(time_format)

    metrics = (
        events_df.groupby(["repo", "time_period"])
        .agg(
            stars=("type", lambda x: (x == "WatchEvent").sum()),
            issues=("type", lambda x: x.isin(["IssuesEvent", "IssueEvent"]).sum()),
            issue_comments=("type", lambda x: (x == "IssueCommentEvent").sum()),
        )
        .reset_index()
        .rename(columns={"repo": "repo_name", "time_period": TIME_KEY})
    )

    return metrics


def update_repo_stats(metrics_df: pd.DataFrame, stats_file: str) -> None:
    """Update repository statistics file with event metrics."""
    if not os.path.exists(stats_file):
        logging.error(f"Repository stats file not found: {stats_file}")
        return

    stats_df = pd.read_csv(stats_file)
    result_df = pd.merge(stats_df, metrics_df, on=["repo_name", TIME_KEY], how="left")

    metric_cols = ["stars", "issues", "issue_comments"]
    result_df[metric_cols] = result_df[metric_cols].fillna(0).astype(int)

    # Get the existing columns and find the position of "contributors"
    existing_cols = list(stats_df.columns)
    if "contributors" in existing_cols:
        contributor_idx = existing_cols.index("contributors")

        # Create a new column order that inserts metrics after "contributors"
        new_cols = existing_cols.copy()
        for metric in reversed(metric_cols):
            if metric in result_df.columns and metric not in existing_cols:
                new_cols.insert(contributor_idx + 1, metric)

        # Reorder columns
        result_df = result_df[new_cols]
        logging.info(f"Inserted metrics after 'contributors' column")

    result_df.to_csv(stats_file, index=False)
    logging.info(f"Updated {stats_file} with {len(metrics_df)} repo-time combinations")


def add_event_metrics_to_repo_stats(aggregation: str) -> None:
    """
    Add GitHub event metrics to repository statistics files.

    Args:
        aggregation: Either 'week' or 'month' for selecting which stats file to update
    """
    if not os.path.exists(REPO_EVENTS_OUTPUT_FILE):
        logging.error(f"Event data file not found: {REPO_EVENTS_OUTPUT_FILE}")
        return

    events_df = pd.read_csv(REPO_EVENTS_OUTPUT_FILE)
    events_df["created_at"] = pd.to_datetime(events_df["created_at"])

    stats_file = Path(__file__).parent.parent / "data" / f"ts_repos_{aggregation}ly.csv"

    if not os.path.exists(stats_file):
        logging.error(f"Repository stats file not found: {stats_file}")
        return

    logging.info(f"Processing {aggregation}ly repository statistics...")

    # Choose the appropriate time format based on aggregation
    time_format = "%Y-%m" if aggregation == "month" else "%Y-W%W"

    # Compute metrics and update the stats file
    metrics = compute_event_metrics(events_df, time_format)
    update_repo_stats(metrics, stats_file)


def main() -> None:
    """Main entry point."""
    # Set up argument parser
    global TIME_KEY
    parser = argparse.ArgumentParser(
        description="Fetch GitHub events and compute metrics for repositories."
    )
    parser.add_argument(
        "--aggregation",
        choices=["week", "month"],
        default="week",
        help="Aggregate data by week or month (default: week)",
    )
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Force fetching events even if the event file already exists",
    )
    args = parser.parse_args()
    TIME_KEY = args.aggregation

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Check if events file exists and fetch only if needed
    if not os.path.exists(REPO_EVENTS_OUTPUT_FILE) or args.force_fetch:
        if args.force_fetch and os.path.exists(REPO_EVENTS_OUTPUT_FILE):
            logging.info(
                f"Force fetching events (file already exists: {REPO_EVENTS_OUTPUT_FILE})"
            )
        else:
            logging.info(f"Events file not found: {REPO_EVENTS_OUTPUT_FILE}")

        repos = load_repos_with_cursor_adoption()
        fetch_events_from_bigquery(repos, REPO_EVENTS_OUTPUT_FILE)
    else:
        logging.info(f"Using existing events file: {REPO_EVENTS_OUTPUT_FILE}")
        logging.info("Use --force-fetch to fetch events again if needed")

    # Always update metrics
    add_event_metrics_to_repo_stats(args.aggregation)


if __name__ == "__main__":
    main()
