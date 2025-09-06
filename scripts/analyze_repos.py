#!/usr/bin/env python3
"""
Script to analyze repositories cloned by clone_repos.py.

This script:
1. Reads repos.csv file and checks which repositories have been cloned
2. Detects when cursor files were first introduced in each repository
3. Collects a time series of weekly or monthly commit counts and lines added for each repository
"""

import argparse
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
OUTPUT_DIR = Path(__file__).parent.parent / "data"
NUM_PROCESSES = multiprocessing.cpu_count() // 2
REPO_TIMEOUT_SECONDS = 7200  # 2 hours timeout per repository
TIME_KEY = None


def get_time_key(commit_time: datetime, aggregation: str) -> str:
    """
    Get time key based on aggregation level.

    Args:
        commit_time (datetime): Commit timestamp
        aggregation (str): Either 'week' or 'month'

    Returns:
        str: Formatted time key
    """
    if aggregation == "week":
        return commit_time.strftime("%Y-W%W")
    else:  # month
        return commit_time.strftime("%Y-%m")


def get_commit_stats(
    repo_path: Path,
    aggregation: str,
) -> Tuple[Optional[Dict[str, Dict[str, int]]], Optional[List[Dict]]]:
    """
    Get weekly or monthly commit counts and lines added for a repository.

    Args:
        repo_path (Path): Path to the repository
        aggregation (str): Either 'week' or 'month'

    Returns:
        Tuple containing:
            1. dict: Dictionary with time period as key and dict of stats as value
                  or None if failed
            2. list: List of contributor-specific stats dictionaries
                  or None if failed
    """
    try:
        repo = git.Repo(str(repo_path))

        # Checkout HEAD before collecting any metrics
        try:
            logging.info("Checking out HEAD for repo %s", repo_path.name)
            repo.git.checkout("HEAD")
        except git.GitCommandError as e:
            logging.error("Failed to checkout HEAD for %s: %s", repo_path.name, e)
            # Continue anyway to try to collect what we can

        repo_name = repo_path.name.replace("_", "/")

        time_stats = defaultdict(
            lambda: {
                "commits": 0,
                "lines_added": 0,
                "lines_removed": 0,
                "contributors": set(),
                "latest_commit": None,
                "latest_commit_time": None,
                "cursor_commits": 0,
            }
        )

        contributor_stats = []
        contributor_time_stats = defaultdict(
            lambda: defaultdict(
                lambda: {"commits": 0, "lines_added": 0, "lines_removed": 0}
            )
        )

        for commit in repo.iter_commits():
            commit_time = datetime.fromtimestamp(commit.committed_date)
            time_key = get_time_key(commit_time, aggregation)
            author_str = f"{commit.author.name} <{commit.author.email}>"

            time_stats[time_key]["commits"] += 1
            time_stats[time_key]["contributors"].add(author_str)

            if (
                time_stats[time_key]["latest_commit_time"] is None
                or commit_time > time_stats[time_key]["latest_commit_time"]
            ):
                time_stats[time_key]["latest_commit"] = commit.hexsha
                time_stats[time_key]["latest_commit_time"] = commit_time

            contributor_time_stats[time_key][author_str]["commits"] += 1

            try:
                if commit.parents:
                    parent = commit.parents[0]
                    added_lines = 0
                    removed_lines = 0

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

                        if result.stdout.strip():
                            for line in result.stdout.strip().split("\n"):
                                if line.strip():
                                    parts = line.split()
                                    if len(parts) >= 2:
                                        if parts[0] != "-":
                                            try:
                                                added_lines += int(parts[0])
                                            except ValueError:
                                                pass
                                        if parts[1] != "-":
                                            try:
                                                removed_lines += int(parts[1])
                                            except ValueError:
                                                pass

                        logging.debug(
                            "%s: +%d -%d lines",
                            commit.hexsha[:8],
                            added_lines,
                            removed_lines,
                        )
                    except subprocess.SubprocessError as e:
                        logging.error("Diff failed for commit %s: %s", commit.hexsha, e)
                        continue

                    time_stats[time_key]["lines_added"] += added_lines
                    time_stats[time_key]["lines_removed"] += removed_lines
                    contributor_time_stats[time_key][author_str][
                        "lines_added"
                    ] += added_lines
                    contributor_time_stats[time_key][author_str][
                        "lines_removed"
                    ] += removed_lines
            except Exception as e:
                logging.debug("Could not get diff for commit %s: %s", commit.hexsha, e)
                continue

        result = {}
        for time_key, stats in time_stats.items():
            result[time_key] = {
                "latest_commit": stats["latest_commit"],
                "commits": stats["commits"],
                "lines_added": stats["lines_added"],
                "lines_removed": stats["lines_removed"],
                "contributors": len(stats["contributors"]),
                "cursor_commits": stats["cursor_commits"],
            }

        for time_key, authors in contributor_time_stats.items():
            for author, stats in authors.items():
                contributor_stats.append(
                    {
                        "repo_name": repo_name,
                        "author": author,
                        TIME_KEY: time_key,
                        "commits": stats["commits"],
                        "lines_added": stats["lines_added"],
                        "lines_removed": stats["lines_removed"],
                    }
                )

        return result, contributor_stats
    except Exception as e:
        logging.error("Failed to get commit info for %s: %s", repo_path, str(e))
        return None, None


def find_cursor_commits(repo_path: Path) -> List[Dict]:
    """
    Scan all commits in a repository to check for .cursorrules files or cursor/ directories.

    Args:
        repo_path (Path): Path to the repository

    Returns:
        List of dictionaries containing commit information for commits that contain
            .cursorrules files or cursor/ directories
    """
    commits_data = []

    try:
        repo = git.Repo(str(repo_path))

        # Checkout HEAD before analyzing
        try:
            logging.info("Checking out HEAD for repo %s", repo_path.name)
            repo.git.checkout("HEAD")
        except git.GitCommandError as e:
            logging.error("Failed to checkout HEAD for %s: %s", repo_path.name, e)
            # Continue anyway to try to collect what we can

        repo_name = repo_path.name.replace("_", "/")

        for commit in repo.iter_commits():
            commit_time = datetime.fromtimestamp(commit.committed_date)
            author_date = datetime.fromtimestamp(commit.authored_date)

            # Check if this commit contains .cursorrules or cursor/ directory
            has_cursor_files = False

            try:
                # Get the tree for this commit
                tree = commit.tree

                # Check for Cursor related files
                if (
                    ".cursorrules" in tree
                    or ".cursor" in tree
                    or ".cursorignore" in tree
                ):
                    has_cursor_files = True

            except Exception as e:
                logging.debug(
                    "Could not check tree for commit %s: %s", commit.hexsha, e
                )
                continue

            if has_cursor_files:
                commits_data.append(
                    {
                        "repo_name": repo_name,
                        "commit_hash": commit.hexsha,
                        "author": f"{commit.author.name} <{commit.author.email}>",
                        "authored_at": author_date.isoformat(),
                        "committer": f"{commit.committer.name} <{commit.committer.email}>",
                        "committed_at": commit_time.isoformat(),
                        "message": commit.message.split("\n")[0],
                    }
                )

        return sorted(commits_data, key=lambda x: x["authored_at"])
    except Exception as e:
        logging.error("Failed to find cursor commits in %s: %s", repo_path, e)
        return []


def count_cursor_commits_by_time(
    cursor_commits: List[Dict], aggregation: str
) -> Dict[str, int]:
    """
    Count cursor commits by time period.

    Args:
        cursor_commits: List of cursor commit data
        aggregation: Either 'week' or 'month'

    Returns:
        Dictionary mapping time keys to cursor commit counts
    """
    time_counts = defaultdict(int)

    for commit in cursor_commits:
        commit_time = datetime.fromisoformat(commit["committed_at"])
        time_key = get_time_key(commit_time, aggregation)
        time_counts[time_key] += 1

    return time_counts


def process_repository(
    idx: int,
    repo: Dict,
    total_repos: int,
    aggregation: str,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a single repository in a worker process.

    Args:
        idx: Index of the repository in the dataframe
        repo: Repository data from the dataframe
        repo_cursor_files: Mapping of repo names to cursor files
        total_repos: Total number of repositories
        aggregation: Either 'week' or 'month'

    Returns:
        Tuple of (repo_ts, contributor_ts) where:
            repo_ts: List of time period statistics dictionaries
            contributor_ts: List of contributor statistics dictionaries
    """
    repo_name = repo["repo_name"]
    repo_path = CLONE_DIR / repo_name.replace("/", "_")
    repo_ts = []
    contributor_ts = []

    if not repo_path.exists():
        return repo_ts, contributor_ts

    logging.info("Analyzing repository: %s (%d/%d)", repo_name, idx + 1, total_repos)

    weekly_stats, contributor_stats = get_commit_stats(repo_path, aggregation)
    cursor_commits = find_cursor_commits(repo_path)
    cursor_commits_by_time = count_cursor_commits_by_time(cursor_commits, aggregation)

    if weekly_stats:
        for time_key, stats in weekly_stats.items():
            has_cursor_usage = cursor_commits_by_time.get(time_key, 0) > 0

            repo_ts.append(
                {
                    "repo_name": repo_name,
                    TIME_KEY: time_key,
                    "latest_commit": stats["latest_commit"],
                    "cursor": has_cursor_usage,
                    "commits": stats["commits"],
                    "lines_added": stats["lines_added"],
                    "lines_removed": stats["lines_removed"],
                    "contributors": stats["contributors"],
                }
            )

    if contributor_stats:
        contributor_ts.extend(contributor_stats)

    return repo_ts, contributor_ts


def main() -> None:
    """Main function to analyze repositories, get commit stats and cursor adoption dates."""
    global TIME_KEY
    parser = argparse.ArgumentParser(
        description="Analyze repository commit history with weekly or monthly aggregation."
    )
    parser.add_argument(
        "--aggregation",
        choices=["week", "month"],
        default="week",
        help="Aggregate data by week or month (default: week)",
    )
    args = parser.parse_args()
    TIME_KEY = args.aggregation

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
    except Exception as e:
        logging.error("Failed to read CSV files: %s", e)
        return

    total_repos = len(repos_df)
    args_list = [
        (idx, repo, total_repos, args.aggregation) for idx, repo in repos_df.iterrows()
    ]
    random.shuffle(args_list)

    repo_ts = []
    contributor_ts = []

    logging.info("Starting multiprocessing pool with %d workers", NUM_PROCESSES)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        async_results = []
        for process_args in args_list:
            async_results.append(pool.apply_async(process_repository, process_args))

        for idx, async_result in enumerate(async_results):
            repo_name = args_list[idx][1]["repo_name"]
            try:
                (
                    repo_time_series,
                    repo_contributor_ts,
                ) = async_result.get(timeout=REPO_TIMEOUT_SECONDS)
                repo_ts.extend(repo_time_series)
                contributor_ts.extend(repo_contributor_ts)
            except multiprocessing.TimeoutError:
                logging.error(
                    "Repository %s processing timed out after %d seconds",
                    repo_name,
                    REPO_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logging.error("Error processing repository %s: %s", repo_name, str(e))

    logging.info("Finished processing %d repos", total_repos)

    # Set output paths based on aggregation level
    output_suffix = "_monthly.csv" if args.aggregation == "month" else "_weekly.csv"
    ts_output = OUTPUT_DIR / f"ts_repos{output_suffix}"
    contributor_output = OUTPUT_DIR / f"ts_contributors{output_suffix}"

    if repo_ts:
        ts_df = pd.DataFrame(repo_ts)
        ts_df.to_csv(ts_output, index=False)
        logging.info("Saved time series data to %s", ts_output)

    if contributor_ts:
        contributor_df = pd.DataFrame(contributor_ts)
        contributor_df.to_csv(contributor_output, index=False)
        logging.info("Saved contributor time series data to %s", contributor_output)


if __name__ == "__main__":
    main()
