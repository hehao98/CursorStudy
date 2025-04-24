#!/usr/bin/env python3
"""
Script to analyze control repositories matched in matching.csv.

This script:
1. Reads matching.csv file to identify control repositories
2. Collects a time series of weekly or monthly commit counts and lines added for each control repository
3. Generates repo_ts_control.csv with the same format as repo_ts.csv
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
from typing import Dict, List, Optional

import git
import pandas as pd

# Constants
MATCHING_CSV = Path(__file__).parent.parent / "data" / "matching.csv"
CONTROL_CLONE_DIR = Path(__file__).parent.parent.parent / "ControlRepos"
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
) -> Optional[Dict[str, Dict[str, int]]]:
    """
    Get weekly or monthly commit counts and lines added for a repository.

    Args:
        repo_path (Path): Path to the repository
        aggregation (str): Either 'week' or 'month'

    Returns:
        dict: Dictionary with time period as key and dict of stats as value
              or None if failed
    """
    try:
        repo = git.Repo(str(repo_path))
        repo_name = repo_path.name.replace("_", "/")

        time_stats = defaultdict(
            lambda: {
                "commits": 0,
                "lines_added": 0,
                "contributors": set(),
                "latest_commit": None,
                "latest_commit_time": None,
            }
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

            try:
                if commit.parents:
                    parent = commit.parents[0]
                    added_lines = 0

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
                                    if len(parts) >= 2 and parts[0] != "-":
                                        try:
                                            added_lines += int(parts[0])
                                        except ValueError:
                                            pass

                        logging.debug("%s: +%d lines", commit.hexsha[:8], added_lines)
                    except subprocess.SubprocessError as e:
                        logging.error("Diff failed for commit %s: %s", commit.hexsha, e)
                        continue

                    time_stats[time_key]["lines_added"] += added_lines
            except Exception as e:
                logging.debug("Could not get diff for commit %s: %s", commit.hexsha, e)
                continue

        result = {}
        for time_key, stats in time_stats.items():
            result[time_key] = {
                "latest_commit": stats["latest_commit"],
                "commits": stats["commits"],
                "lines_added": stats["lines_added"],
                "contributors": len(stats["contributors"]),
            }

        return result
    except Exception as e:
        logging.error("Failed to get commit info for %s: %s", repo_path, str(e))
        return None


def process_repository(
    idx: int,
    repo_name: str,
    total_repos: int,
    aggregation: str,
) -> List[Dict]:
    """
    Process a single repository in a worker process.

    Args:
        idx: Index of the repository in the list
        repo_name: Name of the repository
        total_repos: Total number of repositories
        aggregation: Either 'week' or 'month'

    Returns:
        List of time period statistics dictionaries
    """
    repo_path = CONTROL_CLONE_DIR / repo_name.replace("/", "_")
    repo_ts = []

    if not repo_path.exists():
        return repo_ts

    logging.info(
        "Analyzing control repository: %s (%d/%d)", repo_name, idx + 1, total_repos
    )

    weekly_stats = get_commit_stats(repo_path, aggregation)
    if weekly_stats:
        for time_key, stats in weekly_stats.items():
            repo_ts.append(
                {
                    "repo_name": repo_name,
                    TIME_KEY: time_key,
                    "latest_commit": stats["latest_commit"],
                    "commits": stats["commits"],
                    "lines_added": stats["lines_added"],
                    "contributors": stats["contributors"],
                }
            )

    return repo_ts


def main() -> None:
    """Main function to analyze control repositories and get commit stats."""
    global TIME_KEY
    parser = argparse.ArgumentParser(
        description="Analyze control repository commit history with weekly or monthly aggregation."
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

    if not CONTROL_CLONE_DIR.exists():
        logging.error("Control clone directory %s does not exist", CONTROL_CLONE_DIR)
        return

    try:
        matching_df = pd.read_csv(MATCHING_CSV)
        logging.info("Read matching data for %d repositories", len(matching_df))
    except Exception as e:
        logging.error("Failed to read matching CSV file: %s", e)
        return

    # Extract control repositories from the matching.csv file
    control_repos = []
    for i in range(1, 4):  # matched_control_1, matched_control_2, matched_control_3
        col_name = f"matched_control_{i}"
        if col_name in matching_df.columns:
            control_repos.extend(
                [
                    repo
                    for repo in matching_df[col_name].dropna().tolist()
                    if isinstance(repo, str)
                ]
            )

    # Remove duplicates
    control_repos = list(set(control_repos))
    logging.info("Found %d unique control repositories", len(control_repos))

    total_repos = len(control_repos)
    args_list = [
        (idx, repo_name, total_repos, args.aggregation)
        for idx, repo_name in enumerate(control_repos)
    ]
    random.shuffle(args_list)

    repo_ts = []

    logging.info("Starting multiprocessing pool with %d workers", NUM_PROCESSES)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        async_results = []
        for process_args in args_list:
            async_results.append(pool.apply_async(process_repository, process_args))

        for idx, async_result in enumerate(async_results):
            repo_name = args_list[idx][1]
            try:
                repo_time_series = async_result.get(timeout=REPO_TIMEOUT_SECONDS)
                repo_ts.extend(repo_time_series)
            except multiprocessing.TimeoutError:
                logging.error(
                    "Repository %s processing timed out after %d seconds",
                    repo_name,
                    REPO_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logging.error("Error processing repository %s: %s", repo_name, str(e))

    logging.info("Finished processing %d control repos", total_repos)

    # Set output paths based on aggregation level
    ts_output = OUTPUT_DIR / f"ts_repos_control_{args.aggregation}ly.csv"

    if repo_ts:
        ts_df = pd.DataFrame(repo_ts)
        ts_df.to_csv(ts_output, index=False)
        logging.info("Saved control repo time series data to %s", ts_output)


if __name__ == "__main__":
    main()
