#!/usr/bin/env python3
"""
Script to analyze repositories cloned by clone_repos.py.

This script:
1. Reads repos.csv file and checks which repositories have been cloned
2. Detects when cursor files were first introduced in each repository
3. Collects a time series of monthly commit counts and lines added for each repository
"""

import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import git
import pandas as pd

# Constants
REPOS_CSV = Path(__file__).parent.parent / "data" / "repos.csv"
CURSOR_FILES_CSV = Path(__file__).parent.parent / "data" / "cursor_files.csv"
CLONE_DIR = Path(__file__).parent.parent.parent / "CursorRepos"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "repo_ts.csv"


def get_monthly_commit_stats(repo_path: Path) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Get monthly commit counts and lines added for a repository.

    Args:
        repo_path (Path): Path to the repository

    Returns:
        dict: Dictionary with month as key (YYYY-MM format) and dict of stats as value
              or None if failed
    """
    try:
        # Open the repository
        repo = git.Repo(str(repo_path))

        # Initialize monthly stats counter
        monthly_stats = defaultdict(lambda: {"commits": 0, "lines_added": 0})

        # Iterate through all commits
        for commit in repo.iter_commits():
            # Get commit time and convert to datetime
            commit_time = datetime.fromtimestamp(commit.committed_date)
            # Format as YYYY-MM
            month_key = commit_time.strftime("%Y-%m")
            # Increment commit counter for this month
            monthly_stats[month_key]["commits"] += 1

            # Count lines added in this commit
            try:
                if commit.parents:
                    parent = commit.parents[0]
                    diff = parent.diff(commit)
                    for diff_item in diff:
                        # Count added lines from the diff stats
                        if hasattr(diff_item, "diff"):
                            diff_text = diff_item.diff.decode("utf-8", errors="replace")
                            # Count additions (lines starting with '+' but not '+++')
                            added_lines = sum(
                                1
                                for line in diff_text.splitlines()
                                if line.startswith("+") and not line.startswith("+++")
                            )
                            monthly_stats[month_key]["lines_added"] += added_lines
            except Exception as e:
                logging.debug("Could not get diff for commit %s: %s", commit.hexsha, e)
                continue

        return dict(monthly_stats)
    except Exception as e:
        logging.error("Failed to get commit info for %s: %s", repo_path, str(e))
        return None


def find_cursor_file_introduction(
    repo_path: Path, cursor_files: List[str]
) -> Optional[datetime]:
    """
    Find when cursor files were first introduced in a repository.

    Args:
        repo_path (Path): Path to the repository
        cursor_files (list): List of cursor file paths to look for

    Returns:
        datetime or None: Timestamp of first cursor file introduction or None if not found
    """
    try:
        repo = git.Repo(str(repo_path))
        first_introduction = None

        for file_path in cursor_files:
            # Skip if file doesn't exist in the repo
            if not (repo_path / file_path).exists():
                continue

            try:
                # Use git log to find the earliest commit for this file
                commits = list(repo.iter_commits(paths=file_path))
                if commits:
                    # Last commit in the list is the oldest
                    oldest_commit = commits[-1]
                    commit_time = datetime.fromtimestamp(oldest_commit.committed_date)

                    # Update the first introduction time if this is earlier
                    if first_introduction is None or commit_time < first_introduction:
                        first_introduction = commit_time
            except Exception as e:
                logging.debug("Could not find introduction of %s: %s", file_path, e)
                continue

        return first_introduction
    except Exception as e:
        logging.error("Failed to analyze cursor files in %s: %s", repo_path, e)
        return None


def main() -> None:
    """Main function to analyze repositories, get commit stats and cursor adoption dates."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not CLONE_DIR.exists():
        logging.error("Clone directory %s does not exist", CLONE_DIR)
        return

    # Read the CSV files
    try:
        repos_df = pd.read_csv(REPOS_CSV)
        cursor_files_df = pd.read_csv(CURSOR_FILES_CSV)
        logging.info(
            "Read %d repositories and %d cursor files",
            len(repos_df),
            len(cursor_files_df),
        )
    except Exception as e:
        logging.error("Failed to read CSV files: %s", e)
        return

    # Create mapping of repo_name to cursor files
    repo_cursor_files = defaultdict(list)
    for _, row in cursor_files_df.iterrows():
        repo_cursor_files[row["repo_name"]].append(row["file_path"])

    # Store results
    repo_ts = []
    adoption_dates = {}
    total_repos = len(repos_df)

    for idx, repo in repos_df.iterrows():
        repo_name = repo["repo_name"]
        repo_path = CLONE_DIR / repo_name.replace("/", "_")

        # Skip if repository is not cloned
        if not repo_path.exists():
            continue

        logging.info(
            "Analyzing repository: %s (%d/%d)", repo_name, idx + 1, total_repos
        )

        # Find cursor file introduction if this repo has cursor files
        if repo_name in repo_cursor_files:
            cursor_files = repo_cursor_files[repo_name]
            introduction_date = find_cursor_file_introduction(repo_path, cursor_files)
            if introduction_date:
                adoption_dates[repo_name] = introduction_date.isoformat()
                logging.info("Found cursor adoption date: %s", introduction_date)

        # Get monthly commit stats
        monthly_stats = get_monthly_commit_stats(repo_path)
        if monthly_stats:
            # Add repo data to each month entry
            for month, stats in monthly_stats.items():
                repo_ts.append(
                    {
                        "repo_name": repo_name,
                        "month": month,
                        "commits": stats["commits"],
                        "lines_added": stats["lines_added"],
                    }
                )

        # Log progress periodically
        if (idx + 1) % 10 == 0:
            logging.info("Progress: %d/%d repositories processed", idx + 1, total_repos)

    # Save time series data
    if repo_ts:
        ts_df = pd.DataFrame(repo_ts)
        ts_df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Saved time series data to %s", OUTPUT_FILE)

    # Update repos CSV with cursor adoption dates
    if adoption_dates:
        repos_df["repo_cursor_adoption"] = repos_df["repo_name"].map(adoption_dates)
        repos_df.to_csv(REPOS_CSV, index=False)
        logging.info(
            "Updated %s with cursor adoption dates for %d repositories",
            REPOS_CSV,
            len(adoption_dates),
        )


if __name__ == "__main__":
    main()
