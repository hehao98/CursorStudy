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
MATCHING_CSV = Path(__file__).parent.parent / "data" / "matching.csv"
REPO_EVENTS_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "repo_events.csv"
REPO_EVENTS_CONTROL_OUTPUT_FILE = (
    Path(__file__).parent.parent / "data" / "repo_events_control.csv"
)
TIME_KEY = None
REPO_QUERY = """
SELECT type, created_at, repo.name as repo, actor.login as actor
FROM `githubarchive.day.20*`
WHERE repo.name IN UNNEST(@repos)
ORDER BY repo, created_at
"""

DYNAMIC_METRICS = [
    "commits",
    "lines_added",
    "lines_removed",
    "contributors",
    "stars",
    "issues",
    "issue_comments",
    "cursor_commits",
]
ACCUMULATIVE_METRICS = [
    "ncloc",
    "bugs",
    "vulnerabilities",
    "code_smells",
    "duplicated_lines_density",
    "comment_lines_density",
    "cognitive_complexity",
    "technical_debt",
]


def pad_missing_periods(
    ts_df: pd.DataFrame, group_columns: List[str], time_col: str = "week"
) -> pd.DataFrame:
    """
    Pad missing time periods (weeks or months) in time series data.

    Args:
        ts_df: DataFrame containing time series data
        group_columns: Columns to group by
        time_col: Time column name ('week' or 'month')
    Returns:
        DataFrame with padded time periods for all entities
    """
    padded_ts_dfs = []
    for group_values, group_data in ts_df.groupby(group_columns):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        periods = sorted(group_data[time_col].unique())

        if len(periods) <= 1:
            padded_ts_dfs.append(group_data)
            continue

        # Parse period format strings to dates
        if time_col == "week":
            # Convert format like "2023-W01" to datetime objects
            start_date = pd.to_datetime(periods[0] + "-1", format="%Y-W%W-%w")
            end_date = pd.to_datetime(periods[-1] + "-1", format="%Y-W%W-%w")
            freq = "W"
            date_format = "%Y-W%W"
        else:  # month
            # Convert format like "2023-01" to datetime objects
            start_date = pd.to_datetime(periods[0] + "-01", format="%Y-%m-%d")
            end_date = pd.to_datetime(periods[-1] + "-01", format="%Y-%m-%d")
            freq = "MS"  # Month start frequency
            date_format = "%Y-%m"

        # Determine all periods that should exist in the date range
        all_periods = (
            pd.date_range(
                start=start_date,
                end=end_date,
                freq=freq,
            )
            .strftime(date_format)
            .tolist()
        )

        # Create a dataframe with all periods and the group values
        full_periods_df = pd.DataFrame({time_col: all_periods})
        for i, col in enumerate(group_columns):
            full_periods_df[col] = group_values[i]

        # Merge with existing data and fill missing values
        merged_df = pd.merge(
            full_periods_df, group_data, on=group_columns + [time_col], how="left"
        )

        # Fill dynamic metrics with zeros
        for col in DYNAMIC_METRICS:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)

        # Forward-fill accumulative metrics from previous values
        for col in ACCUMULATIVE_METRICS:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].ffill()

        # Handle repository age as a special accumulative metric
        if "age" in merged_df.columns:
            # Sort by time column
            merged_df = merged_df.sort_values(time_col)

            # Find the first non-null age value
            first_age = (
                merged_df["age"].dropna().iloc[0]
                if not merged_df["age"].dropna().empty
                else 0
            )

            # Calculate the period index for each row
            merged_df["_period_idx"] = range(len(merged_df))

            # Calculate the first period with data
            first_period_idx = merged_df.loc[
                merged_df["age"].notna(), "_period_idx"
            ].min()

            # Calculate age as first_age + (current_period - first_period)
            merged_df["age"] = (
                first_age + (merged_df["_period_idx"] - first_period_idx) * 30
            )

            # Drop temporary column
            merged_df = merged_df.drop(columns=["_period_idx"])

        padded_ts_dfs.append(merged_df)

    return pd.concat(padded_ts_dfs, ignore_index=True)


def load_repos_with_cursor_adoption() -> List[str]:
    """Load repos with Cursor adoption dates from CSV."""
    df = pd.read_csv(REPOS_CSV)
    cursor_repos = df[df["repo_cursor_adoption"].notna()]
    repos = cursor_repos["repo_name"].tolist()
    logging.info(f"Loaded {len(repos)} repos with Cursor adoption dates")
    return repos


def load_control_repos() -> List[str]:
    """Load control repositories from matching CSV file."""
    if not os.path.exists(MATCHING_CSV):
        logging.error(f"Matching CSV file not found: {MATCHING_CSV}")
        return []

    df = pd.read_csv(MATCHING_CSV)
    control_repos = []
    for i in range(1, 4):  # matched_control_1, matched_control_2, matched_control_3
        col_name = f"matched_control_{i}"
        if col_name in df.columns:
            control_repos.extend(
                [
                    repo
                    for repo in df[col_name].dropna().tolist()
                    if isinstance(repo, str)
                ]
            )

    # Remove duplicates
    control_repos = list(set(control_repos))
    logging.info(f"Loaded {len(control_repos)} control repositories")
    return control_repos


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
        logging.warning("No repos found to fetch events for")
        return

    logging.info(f"Preparing to fetch events for {len(repos)} repos")
    bytes_processed, cost = estimate_query_cost(repos)
    logging.info(f"Estimated query size: {format_bytes(bytes_processed)}")
    logging.info(f"Estimated cost: ${cost:.4f} USD")

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
    # Format time periods
    events_df["time_period"] = events_df["created_at"].dt.strftime(time_format)

    # Get first event date for each repository (make timezone naive)
    repo_first_dates = events_df.groupby("repo")["created_at"].min().reset_index()
    repo_first_dates["created_at"] = repo_first_dates["created_at"].dt.tz_localize(None)

    # Calculate base metrics
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

    # Convert time periods to actual dates
    if time_format == "%Y-%m":
        metrics["period_date"] = pd.to_datetime(metrics[TIME_KEY] + "-01")
    else:  # Weekly format
        metrics["period_date"] = pd.to_datetime(
            metrics[TIME_KEY]
            .str.replace("W", "")
            .apply(lambda x: f"{x.split('-')[0]}-{int(x.split('-')[1]):02d}-01")
        )

    # Add first event date to metrics
    metrics = pd.merge(
        metrics,
        repo_first_dates.rename(
            columns={"repo": "repo_name", "created_at": "first_date"}
        ),
        on="repo_name",
        how="left",
    )

    # Calculate repository age (days since first event)
    metrics["age"] = (metrics["period_date"] - metrics["first_date"]).dt.days

    # Remove temporary columns
    metrics = metrics.drop(["period_date", "first_date"], axis=1)

    logging.info(metrics)
    return metrics


def update_repo_stats(
    metrics_df: pd.DataFrame, stats_file: str, aggregation: str
) -> None:
    """Update repository statistics file with event metrics."""
    if not os.path.exists(stats_file):
        logging.error(f"Repository stats file not found: {stats_file}")
        return

    metric_cols = ["stars", "issues", "issue_comments", "age"]
    stats_df = pd.read_csv(stats_file).drop(columns=metric_cols, errors="ignore")

    stats_df = pad_missing_periods(
        stats_df, group_columns=["repo_name"], time_col=aggregation
    )

    result_df = pd.merge(stats_df, metrics_df, on=["repo_name", TIME_KEY], how="left")
    for col in metric_cols:
        result_df[col] = result_df[col].fillna(0).astype(int)
    result_df["age"] = result_df["age"].map(lambda x: 0 if x < 0 else x)

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


def add_event_metrics_to_repo_stats(aggregation: str, control: bool = False) -> None:
    """
    Add GitHub event metrics to repository statistics files.

    Args:
        aggregation: Either 'week' or 'month' for selecting which stats file to update
        control: Whether to update control repository stats (default: False)
    """
    events_file = (
        REPO_EVENTS_CONTROL_OUTPUT_FILE if control else REPO_EVENTS_OUTPUT_FILE
    )
    if not os.path.exists(events_file):
        logging.error(f"Event data file not found: {events_file}")
        return

    events_df = pd.read_csv(events_file)
    events_df["created_at"] = pd.to_datetime(events_df["created_at"])

    # Set the appropriate stats file path based on whether we're handling control repos
    control_suffix = "_control" if control else ""
    stats_file = (
        Path(__file__).parent.parent
        / "data"
        / f"ts_repos{control_suffix}_{aggregation}ly.csv"
    )

    if not os.path.exists(stats_file):
        logging.error(f"Repository stats file not found: {stats_file}")
        return

    logging.info(
        f"Processing {aggregation}ly repository{'_control' if control else ''} statistics..."
    )

    # Choose the appropriate time format based on aggregation
    time_format = "%Y-%m" if aggregation == "month" else "%Y-W%W"

    # Compute metrics and update the stats file
    metrics = compute_event_metrics(events_df, time_format)
    update_repo_stats(metrics, stats_file, aggregation)


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
    parser.add_argument(
        "--control",
        action="store_true",
        help="Process control repositories instead of cursor repositories",
    )
    args = parser.parse_args()
    TIME_KEY = args.aggregation

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Determine which repositories to fetch and which output file to use
    events_file = (
        REPO_EVENTS_CONTROL_OUTPUT_FILE if args.control else REPO_EVENTS_OUTPUT_FILE
    )

    # Check if events file exists and fetch only if needed
    if not os.path.exists(events_file) or args.force_fetch:
        if args.force_fetch and os.path.exists(events_file):
            logging.info(f"Force fetching events (file already exists: {events_file})")
        else:
            logging.info(f"Events file not found: {events_file}")

        # Load appropriate repositories based on --control flag
        if args.control:
            repos = load_control_repos()
        else:
            repos = load_repos_with_cursor_adoption()

        fetch_events_from_bigquery(repos, events_file)
    else:
        logging.info(f"Using existing events file: {events_file}")
        logging.info("Use --force-fetch to fetch events again if needed")

    # Always update metrics
    add_event_metrics_to_repo_stats(args.aggregation, args.control)


if __name__ == "__main__":
    main()
