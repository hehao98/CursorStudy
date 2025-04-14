#!/usr/bin/env python3
"""
Script to analyze repositories cloned by clone_repos.py.

This script:
1. Reads repos.csv file and checks which repositories have been cloned
2. Detects when cursor files were first introduced in each repository
3. Collects a time series of weekly commit counts and lines added for each repository
"""

import logging
import multiprocessing
import random
import subprocess
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
CURSOR_COMMITS_CSV = Path(__file__).parent.parent / "data" / "cursor_commits.csv"
CLONE_DIR = Path(__file__).parent.parent.parent / "CursorRepos"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "ts_repos.csv"
CONTRIBUTOR_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "ts_contributors.csv"
NUM_PROCESSES = multiprocessing.cpu_count() // 2
REPO_TIMEOUT_SECONDS = 7200  # 2 hours timeout per repository


def get_weekly_commit_stats(
    repo_path: Path,
) -> Tuple[Optional[Dict[str, Dict[str, int]]], Optional[List[Dict]]]:
    """
    Get weekly commit counts and lines added for a repository.

    Args:
        repo_path (Path): Path to the repository

    Returns:
        Tuple containing:
            1. dict: Dictionary with week as key (YYYY-WXX format) and dict of stats as value
                  or None if failed
            2. list: List of contributor-specific stats dictionaries
                  or None if failed
    """
    try:
        repo = git.Repo(str(repo_path))
        repo_name = repo_path.name.replace("_", "/")

        weekly_stats = defaultdict(
            lambda: {
                "commits": 0,
                "lines_added": 0,
                "contributors": set(),
                "latest_commit": None,
                "latest_commit_time": None,
            }
        )

        contributor_stats = []

        contributor_weekly_stats = defaultdict(
            lambda: defaultdict(lambda: {"commits": 0, "lines_added": 0})
        )

        for commit in repo.iter_commits():
            commit_time = datetime.fromtimestamp(commit.committed_date)
            week_key = commit_time.strftime("%Y-W%W")
            author_str = f"{commit.author.name} <{commit.author.email}>"

            weekly_stats[week_key]["commits"] += 1
            weekly_stats[week_key]["contributors"].add(author_str)

            # Update latest commit if this is the first one seen this week or has a later timestamp
            if (
                weekly_stats[week_key]["latest_commit_time"] is None
                or commit_time > weekly_stats[week_key]["latest_commit_time"]
            ):
                weekly_stats[week_key]["latest_commit"] = commit.hexsha
                weekly_stats[week_key]["latest_commit_time"] = commit_time

            contributor_weekly_stats[week_key][author_str]["commits"] += 1

            try:
                if commit.parents:
                    parent = commit.parents[0]
                    added_lines = 0

                    # Use git CLI to get lines added as GitPython is not working
                    try:
                        cmd = [
                            "git",
                            "-C",
                            str(repo_path),
                            "diff",
                            "--numstat",
                            f"{parent.hexsha}..{commit.hexsha}",
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, check=True
                        )

                        # Parse the numstat output: each line has format "added deleted filename"
                        if result.stdout.strip():
                            for line in result.stdout.strip().split("\n"):
                                if line.strip():
                                    parts = line.split()
                                    if (
                                        len(parts) >= 2 and parts[0] != "-"
                                    ):  # Skip binary files (marked with -)
                                        try:
                                            added_lines += int(parts[0])
                                        except ValueError:
                                            pass  # Skip if conversion fails

                        logging.debug("%s: +%d lines", commit.hexsha[:8], added_lines)
                    except subprocess.SubprocessError as e:
                        logging.error("Diff failed for commit %s: %s", commit.hexsha, e)
                        continue

                    weekly_stats[week_key]["lines_added"] += added_lines
                    contributor_weekly_stats[week_key][author_str][
                        "lines_added"
                    ] += added_lines
            except Exception as e:
                logging.debug("Could not get diff for commit %s: %s", commit.hexsha, e)
                continue

        result = {}
        for week, stats in weekly_stats.items():
            result[week] = {
                "latest_commit": stats["latest_commit"],
                "commits": stats["commits"],
                "lines_added": stats["lines_added"],
                "contributors": len(stats["contributors"]),
            }

        for week, authors in contributor_weekly_stats.items():
            for author, stats in authors.items():
                contributor_stats.append(
                    {
                        "repo_name": repo_name,
                        "author": author,
                        "week": week,
                        "commits": stats["commits"],
                        "lines_added": stats["lines_added"],
                    }
                )

        return result, contributor_stats
    except Exception as e:
        logging.error("Failed to get commit info for %s: %s", repo_path, str(e))
        return None, None


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


def find_cursor_file_commits(repo_path: Path, cursor_files: List[str]) -> List[Dict]:
    """
    Find all commits that modified cursor files in a repository.

    Args:
        repo_path (Path): Path to the repository
        cursor_files (list): List of cursor file paths to check

    Returns:
        List of dictionaries containing commit information
    """
    commits_data = []

    try:
        repo = git.Repo(str(repo_path))
        repo_name = repo_path.name.replace("_", "/")

        for file_path in cursor_files:
            # Skip if file doesn't exist in the repo
            if not (repo_path / file_path).exists():
                continue

            try:
                # Get all commits that modified this file
                for commit in repo.iter_commits(paths=file_path):
                    commit_time = datetime.fromtimestamp(commit.committed_date)
                    author_date = datetime.fromtimestamp(commit.authored_date)

                    commits_data.append(
                        {
                            "repo_name": repo_name,
                            "commit_hash": commit.hexsha,
                            "cursor_file": file_path,
                            "author": f"{commit.author.name} <{commit.author.email}>",
                            "authored_at": author_date.isoformat(),
                            "committer": f"{commit.committer.name} <{commit.committer.email}>",
                            "committed_at": commit_time.isoformat(),
                            "message": commit.message,
                        }
                    )
            except Exception as e:
                logging.error(f"Error processing commits for {file_path}: {e}")
                continue

        return commits_data
    except Exception as e:
        logging.error(f"Failed to find cursor file commits in {repo_path}: {e}")
        return []


def process_repository(
    idx: int, repo: Dict, repo_cursor_files: Dict[str, List[str]], total_repos: int
) -> Tuple[List[Dict], Dict[str, str], List[Dict], List[Dict]]:
    """
    Process a single repository in a worker process.

    Args:
        idx: Index of the repository in the dataframe
        repo: Repository data from the dataframe
        repo_cursor_files: Mapping of repo names to cursor files
        total_repos: Total number of repositories

    Returns:
        Tuple of (repo_ts, adoption_date, contributor_ts, cursor_commits) where:
            repo_ts: List of weekly statistics dictionaries
            adoption_date: Dictionary mapping repo_name to cursor adoption date
            contributor_ts: List of contributor statistics dictionaries
            cursor_commits: List of commits modifying cursor files
    """
    repo_name = repo["repo_name"]
    repo_path = CLONE_DIR / repo_name.replace("/", "_")
    repo_ts = []
    contributor_ts = []
    cursor_commits = []
    adoption_date = {}

    if not repo_path.exists():
        return repo_ts, adoption_date, contributor_ts, cursor_commits

    logging.info("Analyzing repository: %s (%d/%d)", repo_name, idx + 1, total_repos)

    if repo_name in repo_cursor_files:
        cursor_files = repo_cursor_files[repo_name]
        introduction_date = find_cursor_file_introduction(repo_path, cursor_files)
        if introduction_date:
            adoption_date[repo_name] = introduction_date.isoformat()
            logging.info("Found cursor adoption date: %s", introduction_date)

        # Find commits that modified cursor files
        cursor_commits = find_cursor_file_commits(repo_path, cursor_files)
        if cursor_commits:
            logging.info("Found %d commits modifying cursor files", len(cursor_commits))

    weekly_stats, contributor_stats = get_weekly_commit_stats(repo_path)
    if weekly_stats:
        for week, stats in weekly_stats.items():
            repo_ts.append(
                {
                    "repo_name": repo_name,
                    "week": week,
                    "commits": stats["commits"],
                    "lines_added": stats["lines_added"],
                    "contributors": stats["contributors"],
                }
            )

    if contributor_stats:
        contributor_ts.extend(contributor_stats)

    return repo_ts, adoption_date, contributor_ts, cursor_commits


def main() -> None:
    """Main function to analyze repositories, get commit stats and cursor adoption dates."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    multiprocessing.freeze_support()

    if not CLONE_DIR.exists():
        logging.error("Clone directory %s does not exist", CLONE_DIR)
        return

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

    repo_cursor_files = defaultdict(list)
    for _, row in cursor_files_df.iterrows():
        repo_cursor_files[row["repo_name"]].append(row["file_path"])

    total_repos = len(repos_df)
    args_list = [
        (idx, repo, repo_cursor_files, total_repos) for idx, repo in repos_df.iterrows()
    ]
    random.shuffle(args_list)  # Hope to have equal load per process

    repo_ts = []
    contributor_ts = []
    cursor_commits = []
    adoption_dates = {}

    logging.info("Starting multiprocessing pool with %d workers", NUM_PROCESSES)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # Create async results
        async_results = []
        for args in args_list:
            async_results.append(pool.apply_async(process_repository, args))

        # Process results with timeout
        for idx, async_result in enumerate(async_results):
            repo_name = args_list[idx][1]["repo_name"]
            try:
                (
                    repo_time_series,
                    repo_adoption_date,
                    repo_contributor_ts,
                    repo_cursor_commits,
                ) = async_result.get(timeout=REPO_TIMEOUT_SECONDS)
                repo_ts.extend(repo_time_series)
                contributor_ts.extend(repo_contributor_ts)
                cursor_commits.extend(repo_cursor_commits)
                adoption_dates.update(repo_adoption_date)
            except multiprocessing.TimeoutError:
                logging.error(
                    "Repository %s processing timed out after %d seconds",
                    repo_name,
                    REPO_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logging.error("Error processing repository %s: %s", repo_name, str(e))

    logging.info("Finished processing %d repos", total_repos)

    if repo_ts:
        ts_df = pd.DataFrame(repo_ts)
        ts_df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Saved time series data to %s", OUTPUT_FILE)

    if contributor_ts:
        contributor_df = pd.DataFrame(contributor_ts)
        contributor_df.to_csv(CONTRIBUTOR_OUTPUT_FILE, index=False)
        logging.info(
            "Saved contributor time series data to %s", CONTRIBUTOR_OUTPUT_FILE
        )

    if cursor_commits:
        cursor_commits_df = pd.DataFrame(cursor_commits)
        cursor_commits_df.to_csv(CURSOR_COMMITS_CSV, index=False)
        logging.info(
            "Saved %d cursor commits to %s", len(cursor_commits), CURSOR_COMMITS_CSV
        )

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
