"""
Script to collect package.json dependency data on the latest commits of each week.

This script:
1. Reads the weekly repository stats from ts_repos.csv
2. For each repository and week, pulls the package.json file on the latest commit
3. Collects and stores the dependency data (both aggregate weekly values and all raw dependency data for technical lag calculations)
"""

import logging
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import git
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Metrics to collect from SonarQube
METRICS_OF_INTEREST = [
    "num_dependencies",  # technical debt
]

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CLONE_DIR = SCRIPT_DIR / "CursorRepos"
TS_REPOS_CSV = DATA_DIR / "ts_repos.csv"
REPOS_CSV = DATA_DIR / "repos.csv"  # Add path for repos data
WEEKS_INTERVAL = 30  # It may be too costly to analyze all weeks
NUM_PROCESSES = 16  # Number of processes to use for parallel processing


def process_repository(
    ts_df: pd.DataFrame, repos_df: pd.DataFrame, repo_name: str
) -> pd.DataFrame:
    """
    Process a single repository's analysis.

    Args:
        ts_df: Time series dataframe
        repos_df: Repository information dataframe
        repo_name: Name of the repository to process

    Returns:
        pd.DataFrame: Updated time series dataframe for this repository
    """
    project_key = repo_name.replace("/", "_")
    repo_path = CLONE_DIR / repo_name.replace("/", "_")
    if not repo_path.exists():
        logging.warning("Repository %s not found at %s", repo_name, repo_path)
        return ts_df[ts_df["repo_name"] == repo_name]

    # Get adoption week from repos data
    repo_info = repos_df[repos_df["repo_name"] == repo_name]
    if repo_info.empty or pd.isna(repo_info["repo_cursor_adoption"].iloc[0]):
        logging.warning("No adoption found for %s, skipping", repo_name)
        return ts_df[ts_df["repo_name"] == repo_name]

    adoption = repo_info["repo_cursor_adoption"].iloc[0]
    start_week = (adoption - pd.Timedelta(weeks=WEEKS_INTERVAL)).strftime("%Y-W%W")
    end_week = (adoption + pd.Timedelta(weeks=WEEKS_INTERVAL)).strftime("%Y-W%W")
    logging.info("Processing %s from %s to %s", repo_name, start_week, end_week)

    # Get repository data
    repo_df = ts_df[ts_df["repo_name"] == repo_name].copy()

    print(repo_df)


    # Filter weeks before and after adoption
    weeks_in_range = sorted(
        repo_df[(repo_df["week"] >= start_week) & (repo_df["week"] <= end_week)][
            "week"
        ].unique()
    )

    # # Process each week's latest commit in chronological order
    # for week in weeks_in_range:
    #     # Get the index in the repository dataframe
    #     row_idx = repo_df[repo_df["week"] == week].index[0]
    #     commit_hash = repo_df.loc[row_idx, "latest_commit"]
    #
    #     if not commit_hash:
    #         logging.warning("No commit hash for %s at %s", repo_name, week)
    #         continue
    #
    #     if not check_analysis_exists(project_key, week):
    #         logging.info("%s at %s (%s)", repo_name, week, commit_hash[:8])
    #         run_sonar_scan(repo_path, commit_hash, week, project_key)
    #
    #     metrics = get_sonar_metrics(project_key)
    #     if metrics:
    #         for metric, value in metrics.items():
    #             repo_df.loc[row_idx, metric] = value
    #     logging.info("Metrics for %s at %s: %s", repo_name, week, metrics)

    return repo_df



def main() -> None:
    """Main function to run project.json dependency data collection on repositories."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Read repository time series data and adoption data
    ts_df = pd.read_csv(TS_REPOS_CSV)
    repos_df = pd.read_csv(REPOS_CSV)

    # Convert cursor_adoption_week to datetime
    repos_df["repo_cursor_adoption"] = pd.to_datetime(repos_df["repo_cursor_adoption"])

    # Create columns for metrics if they don't exist
    for col in METRICS_OF_INTEREST:
        if col not in ts_df.columns:
            ts_df[col] = None

    # Get unique repository names
    repo_names = ts_df["repo_name"].unique()

    # pull name of first repo as example TODO
    repo_name = repo_names[0]

    # pulling results for repo
    results = process_repository(ts_df, repos_df, repo_name)





if __name__ == "__main__":
    main()
