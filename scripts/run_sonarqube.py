#!/usr/bin/env python3
"""
Script to run SonarQube scanner on the latest commits of each week.

This script:
1. Reads the weekly repository stats from ts_repos.csv
2. For each repository and week, runs SonarQube scanner on the latest commit
3. Collects and stores the analysis results
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import git
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
SONAR_PATH = os.getenv("SONAR_PATH")
SONAR_TOKEN = os.getenv("SONAR_TOKEN")
SONAR_HOST = "http://localhost:9000"

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CLONE_DIR = SCRIPT_DIR.parent.parent / "CursorRepos"
TS_REPOS_CSV = DATA_DIR / "ts_repos.csv"


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
        repo.git.checkout(commit_hash)

        try:
            # Run sonar-scanner
            cmd = [
                SONAR_PATH,
                f"-Dsonar.projectKey={project_key}",
                f"-Dsonar.projectName={project_key}",
                f"-Dsonar.projectVersion={version}",
                "-Dsonar.sources=.",
                f"-Dsonar.host.url={SONAR_HOST}",
                f"-Dsonar.token={SONAR_TOKEN}",
                "-Dsonar.scm.disabled=true",  # Disable SCM to speed up analysis
            ]

            result = subprocess.run(
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


def get_sonar_metrics(project_key: str) -> Optional[Dict]:
    """
    Get metrics from SonarQube API for a project.

    Args:
        project_key: SonarQube project key

    Returns:
        dict: Metrics data or None if request failed
    """
    try:
        url = f"{SONAR_HOST}/api/measures/component"
        headers = {"Authorization": f"Bearer {SONAR_TOKEN}"}
        params = {
            "component": project_key,
            "metricKeys": "bugs,vulnerabilities,code_smells,coverage,duplicated_lines_density",
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        metrics = {}
        for measure in data["component"]["measures"]:
            metrics[measure["metric"]] = float(measure["value"])

        return metrics

    except requests.exceptions.RequestException as e:
        logging.error("Failed to get metrics for %s: %s", project_key, str(e))
        return None


def main() -> None:
    """Main function to run SonarQube analysis on repositories."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not SONAR_PATH or not SONAR_TOKEN:
        logging.error("SONAR_PATH and SONAR_TOKEN must be set in .env file")
        return

    # Read repository time series data
    ts_df = pd.read_csv(TS_REPOS_CSV)

    # Create columns for metrics if they don't exist
    metric_columns = [
        "bugs",
        "vulnerabilities",
        "code_smells",
        "coverage",
        "duplicated_lines_density",
    ]
    for col in metric_columns:
        if col not in ts_df.columns:
            ts_df[col] = None

    # Group by repository name to process each repo's commits
    for repo_name, repo_group in ts_df.groupby("repo_name"):
        repo_path = CLONE_DIR / repo_name.replace("/", "_")
        if not repo_path.exists():
            logging.warning("Repository %s not found at %s", repo_name, repo_path)
            continue

        # Sort weeks chronologically to ensure analysis runs from early to late weeks
        weeks_sorted = sorted(repo_group["week"].unique())

        # Process each week's latest commit in chronological order
        for week in weeks_sorted:
            # Get the index in the original dataframe
            row_idx = ts_df[
                (ts_df["repo_name"] == repo_name) & (ts_df["week"] == week)
            ].index[0]
            commit_hash = ts_df.loc[row_idx, "latest_commit"]

            if not commit_hash:
                logging.warning("No commit hash for %s at %s", repo_name, week)
                continue

            if not check_analysis_exists(repo_name, week):
                logging.info(
                    "Processing %s at %s (%s)", repo_name, week, commit_hash[:8]
                )
                run_sonar_scan(repo_path, commit_hash, week, repo_name)

            metrics = get_sonar_metrics(repo_name)
            if metrics:
                for metric, value in metrics.items():
                    ts_df.loc[row_idx, metric] = value

    # Save updated dataframe
    ts_df.to_csv(TS_REPOS_CSV, index=False)
    logging.info("Updated metrics saved to %s", TS_REPOS_CSV)


if __name__ == "__main__":
    main()
