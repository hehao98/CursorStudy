#!/usr/bin/env python3
"""
Script to run SonarQube scanner on the latest commits of each week or month.

This script:
1. Reads the time series repository stats from ts_repos_weekly.csv or ts_repos_monthly.csv
2. For each repository and time period, runs SonarQube scanner on the latest commit
3. Collects and stores the analysis results

NOTE: Sometimes the analysis results are not immediately available in database,
so you may have to run this script twice in order to fetch all available metrics
"""

import argparse
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

# Constants
SONAR_PATH = os.getenv("SONAR_SCANNER_PATH")
SONAR_TOKEN = os.getenv("SONAR_TOKEN")
SONAR_HOST = os.getenv("SONAR_HOST")

# Time key in the time series dataframe
TIME_KEY = None
TIME_PERIODS_INTERVAL = None

# Metrics to collect from SonarQube
METRICS_OF_INTEREST = [
    "ncloc",
    "bugs",
    "vulnerabilities",
    "code_smells",
    "duplicated_lines_density",
    "comment_lines_density",
    "cognitive_complexity",
    "software_quality_maintainability_remediation_effort",  # technical debt
]

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CLONE_DIR = SCRIPT_DIR.parent.parent / "CursorRepos"
REPOS_CSV = DATA_DIR / "repos.csv"  # Add path for repos data

# Number of processes to use for parallel processing
NUM_PROCESSES = 16

# Taking too long to analyze plus no metrics are able to be collected
REPO_IGNORE = ["meshery/meshery"]


def check_analysis_exists(project_key: str, version: str) -> bool:
    """
    Check if SonarQube analysis already exists for a project and version.

    Args:
        project_key: SonarQube project key
        version: Version identifier of the analysis

    Returns:
        bool: True if analysis exists, False otherwise
    """
    try:
        page = 1
        while True:
            url = f"{SONAR_HOST}/api/project_analyses/search"
            headers = {"Authorization": f"Bearer {SONAR_TOKEN}"}
            params = {
                "project": project_key,
                "category": "VERSION",
                "p": page,
                "ps": 100,  # Page size of 100 analyses
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()
            if "analyses" not in data or not data["analyses"]:
                # No more analyses to check
                break

            for analysis in data["analyses"]:
                if analysis.get("projectVersion") == version:
                    logging.info(
                        "Analysis already exists for %s version %s",
                        project_key,
                        version,
                    )
                    return True

            # Check if we've reached the last page
            if len(data["analyses"]) < 100:
                break

            page += 1

        return False

    except requests.exceptions.RequestException as e:
        logging.error(
            "Failed to check existing analysis for %s: %s", project_key, str(e)
        )
        return False


def run_sonar_scan(
    repo_path: Path, commit_hash: str, version: str, project_key: str
) -> bool:
    """
    Run SonarQube scanner on a specific commit.

    Args:
        repo_path: Path to the repository
        commit_hash: Git commit hash to analyze
        version: Version identifier for the analysis
        project_key: SonarQube project key

    Returns:
        bool: True if scan was successful, False otherwise
    """
    try:
        # Checkout the specific commit
        repo = git.Repo(str(repo_path))
        current = repo.head.commit

        # Force checkout and clean the working directory
        repo.git.reset("--hard")
        repo.git.clean("-fd")
        repo.git.checkout(commit_hash, force=True)

        try:
            # Run sonar-scanner
            cmd = [
                SONAR_PATH,
                f"-Dsonar.projectKey={project_key}",
                f"-Dsonar.projectName={project_key}",
                f"-Dsonar.projectVersion={version}",
                "-Dsonar.sources=.",
                f"-Dsonar.java.binaries=.",  # Fix Java errors, hopefully we find some .class here
                f"-Dsonar.host.url={SONAR_HOST}",
                f"-Dsonar.token={SONAR_TOKEN}",
                "-Dsonar.scm.disabled=true",  # Disable SCM to speed up analysis
            ]

            subprocess.run(
                cmd, cwd=repo_path, capture_output=True, text=True, check=True
            )
            logging.info("SonarQube scan completed for %s at %s", project_key, version)
            return True

        finally:
            # Always return to original commit
            repo.git.checkout(current)

    except subprocess.CalledProcessError as e:
        logging.error(
            "SonarQube scan failed for %s at %s: %s", project_key, version, e.stderr
        )
        return False
    except Exception as e:
        logging.error("Error during scan of %s at %s: %s", project_key, version, str(e))
        return False


def get_sonar_metrics(project_key: str, version: str) -> Optional[Dict]:
    """
    Get metrics from SonarQube API for a project and specific version.

    Args:
        project_key: SonarQube project key
        version: Version identifier of the analysis

    Returns:
        dict: Metrics data or None if request failed
    """
    try:
        # First, find the analysis with the matching version to get its date
        analysis_date = None
        page = 1

        while analysis_date is None:
            url = f"{SONAR_HOST}/api/project_analyses/search"
            headers = {"Authorization": f"Bearer {SONAR_TOKEN}"}
            params = {
                "project": project_key,
                "category": "VERSION",
                "ps": 100,  # Page size
                "p": page,  # Page number
            }

            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()

            data = response.json()

            logging.info("Found %d analyses for %s", len(data["analyses"]), project_key)

            # Find the analysis with matching version to get its date
            if "analyses" in data and data["analyses"]:
                for analysis in data["analyses"]:
                    if analysis.get("projectVersion") == version:
                        analysis_date = analysis.get("date")
                        break

                # If we haven't found the analysis and there are more pages, continue to the next page
                if analysis_date is None and len(data["analyses"]) == 100:
                    page += 1
                else:
                    break  # No more results or found the analysis
            else:
                break  # No analyses returned

        if not analysis_date:
            logging.warning("No analysis found for %s version %s", project_key, version)
            return None

        # Now get the measures for this specific date using search_history
        url = f"{SONAR_HOST}/api/measures/search_history"
        params = {
            "component": project_key,
            "metrics": ",".join(METRICS_OF_INTEREST),
            "from": analysis_date,
            "to": analysis_date,
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        metrics = {}

        if "measures" in data:
            for measure in data["measures"]:
                if (
                    measure["history"]
                    and len(measure["history"]) > 0
                    and "value" in measure["history"][0]
                ):
                    metrics[measure["metric"]] = float(measure["history"][0]["value"])

        return metrics if metrics else None

    except requests.exceptions.RequestException as e:
        logging.error(
            "Failed to get metrics for %s version %s: %s", project_key, version, str(e)
        )
        return None


def process_repository(
    ts_df: pd.DataFrame, repos_df: pd.DataFrame, repo_name: str, aggregation: str
) -> pd.DataFrame:
    """
    Process a single repository's analysis.

    Args:
        ts_df: Time series dataframe
        repos_df: Repository information dataframe
        repo_name: Name of the repository to process
        aggregation: Either 'week' or 'month'

    Returns:
        pd.DataFrame: Updated time series dataframe for this repository
    """
    project_key = repo_name.replace("/", "_")
    repo_path = CLONE_DIR / repo_name.replace("/", "_")
    if not repo_path.exists():
        logging.warning("Repository %s not found at %s", repo_name, repo_path)
        return ts_df[ts_df["repo_name"] == repo_name]

    # Get adoption time from repos data
    repo_info = repos_df[repos_df["repo_name"] == repo_name]
    if repo_info.empty or pd.isna(repo_info["repo_cursor_adoption"].iloc[0]):
        logging.warning("No adoption found for %s, skipping", repo_name)
        return ts_df[ts_df["repo_name"] == repo_name]

    adoption = repo_info["repo_cursor_adoption"].iloc[0]

    # Determine date range based on aggregation
    if aggregation == "week":
        start_time = (adoption - pd.Timedelta(weeks=TIME_PERIODS_INTERVAL)).strftime(
            "%Y-W%W"
        )
        end_time = (adoption + pd.Timedelta(weeks=TIME_PERIODS_INTERVAL)).strftime(
            "%Y-W%W"
        )
    else:  # month
        start_month = pd.Timestamp(adoption.year, adoption.month, 1) - pd.DateOffset(
            months=TIME_PERIODS_INTERVAL
        )
        end_month = pd.Timestamp(adoption.year, adoption.month, 1) + pd.DateOffset(
            months=TIME_PERIODS_INTERVAL
        )
        start_time = start_month.strftime("%Y-%m")
        end_time = end_month.strftime("%Y-%m")

    logging.info("Processing %s from %s to %s", repo_name, start_time, end_time)

    # Get repository data
    repo_df = ts_df[ts_df["repo_name"] == repo_name].copy()

    # Filter time periods before and after adoption
    time_periods_in_range = sorted(
        repo_df[(repo_df[TIME_KEY] >= start_time) & (repo_df[TIME_KEY] <= end_time)][
            TIME_KEY
        ].unique()
    )

    # Process each time period's latest commit in chronological order
    for time_period in time_periods_in_range:
        # Get the index in the repository dataframe
        row_idx = repo_df[repo_df[TIME_KEY] == time_period].index[0]
        commit_hash = repo_df.loc[row_idx, "latest_commit"]

        if not commit_hash:
            logging.warning("No commit hash for %s at %s", repo_name, time_period)
            continue

        if not check_analysis_exists(project_key, time_period):
            logging.info("%s at %s (%s)", repo_name, time_period, commit_hash[:8])
            run_sonar_scan(repo_path, commit_hash, time_period, project_key)

        metrics = get_sonar_metrics(project_key, time_period)
        if metrics:
            for metric, value in metrics.items():
                repo_df.loc[row_idx, metric] = value
        logging.info("Metrics for %s at %s: %s", repo_name, time_period, metrics)

    return repo_df


def main() -> None:
    """Main function to run SonarQube analysis on repositories."""
    global TIME_PERIODS_INTERVAL, TIME_KEY
    parser = argparse.ArgumentParser(
        description="Run SonarQube analysis on repository commits with weekly or monthly aggregation."
    )
    parser.add_argument(
        "--aggregation",
        choices=["week", "month"],
        default="week",
        help="Aggregate data by week or month (default: week)",
    )
    args = parser.parse_args()
    TIME_KEY = "week" if args.aggregation == "week" else "month"
    if args.aggregation == "week":
        TIME_PERIODS_INTERVAL = 30
    elif args.aggregation == "month":
        TIME_PERIODS_INTERVAL = 6

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not SONAR_PATH or not SONAR_TOKEN:
        logging.error("SONAR_PATH and SONAR_TOKEN must be set in .env file")
        return

    # Set input file path based on aggregation level
    ts_repos_file = DATA_DIR / f"ts_repos_{args.aggregation}ly.csv"

    # Read repository time series data and adoption data
    try:
        ts_df = pd.read_csv(ts_repos_file)
        repos_df = pd.read_csv(REPOS_CSV)
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
        return

    # Convert cursor_adoption_week to datetime
    repos_df["repo_cursor_adoption"] = pd.to_datetime(repos_df["repo_cursor_adoption"])

    # Create columns for metrics if they don't exist
    for col in METRICS_OF_INTEREST:
        if col not in ts_df.columns:
            ts_df[col] = None

    # Get unique repository names
    repo_names = set(ts_df["repo_name"].unique()) - set(REPO_IGNORE)
    with mp.Pool(NUM_PROCESSES) as pool:
        args_list = [
            (ts_df, repos_df, repo_name, args.aggregation) for repo_name in repo_names
        ]
        results = pool.starmap(process_repository, args_list, chunksize=1)

    updated_df = pd.concat(results)
    updated_df.drop(columns=["technical_debt"], inplace=True)
    updated_df.rename(
        columns={
            "software_quality_maintainability_remediation_effort": "technical_debt"
        },
        inplace=True,
    )
    updated_df.sort_values(by=["repo_name", TIME_KEY]).to_csv(
        ts_repos_file, index=False
    )
    logging.info("Updated metrics saved to %s", ts_repos_file)


if __name__ == "__main__":
    main()
