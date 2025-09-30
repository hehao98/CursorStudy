#!/usr/bin/env python3
"""
Compute repository-level metrics for cursor adoption.

This script computes:
1. Number of cursor file changing commits per repository
2. Percentage of cursor adoptor commits per repository

Cursor adoptor commits are defined as commits authored by developers
who have at least one cursor-file changing commit.
"""

import logging
import sys
from typing import Any, Dict, Set, Tuple

import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required data files."""
    logging.info("Loading data files...")
    repos_df = pd.read_csv("data/repos.csv")
    logging.info("Loaded %d repositories", len(repos_df))

    contributors_df = pd.read_csv("data/ts_contributors_monthly.csv")
    logging.info("Loaded %d contributor records", len(contributors_df))

    cursor_commits_df = pd.read_csv("data/cursor_commits.csv")
    logging.info("Loaded %d cursor commits", len(cursor_commits_df))

    return repos_df, contributors_df, cursor_commits_df


def filter_repos_by_stars(repos_df: pd.DataFrame, min_stars: int = 10) -> Set[str]:
    """Filter repositories with at least min_stars GitHub stars."""
    filtered_repos = set(repos_df[repos_df["repo_stars"] >= min_stars]["repo_name"])
    logging.info(
        "Found %d repositories with >= %d stars", len(filtered_repos), min_stars
    )
    return filtered_repos


def get_cursor_developers(cursor_commits_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Get developers who have made cursor-file changing commits per repo."""
    logging.info("Identifying cursor developers...")
    cursor_devs = {}
    for _, row in cursor_commits_df.iterrows():
        repo = row["repo_name"]
        author = row["author"]

        if repo not in cursor_devs:
            cursor_devs[repo] = set()
        cursor_devs[repo].add(author)

    repos_with_cursor = len(cursor_devs)
    total_cursor_devs = sum(len(devs) for devs in cursor_devs.values())
    logging.info(
        "Found %d repositories with cursor commits and %d unique cursor developers",
        repos_with_cursor,
        total_cursor_devs,
    )
    return cursor_devs


def compute_first_cursor_commit_times(
    cursor_commits_df: pd.DataFrame,
) -> Dict[str, str]:
    """Compute first cursor commit authored_at timestamp per repository."""
    logging.info("Computing first cursor commit times...")

    if cursor_commits_df.empty:
        return {}

    if "repo_name" not in cursor_commits_df or "authored_at" not in cursor_commits_df:
        logging.warning("cursor_commits_df missing required columns")
        return {}

    first_times_series = cursor_commits_df.groupby("repo_name")["authored_at"].min()
    first_times: Dict[str, str] = first_times_series.to_dict()

    logging.info(
        "Computed first cursor commit times for %d repositories", len(first_times)
    )
    return first_times


def count_commits_per_developer(
    contributors_df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """Count total commits per developer per repository."""
    logging.info("Aggregating commits per developer...")
    commits_by_repo_dev = {}

    for _, row in contributors_df.iterrows():
        repo = row["repo_name"]
        author = row["author"]
        commits = row["commits"]

        if repo not in commits_by_repo_dev:
            commits_by_repo_dev[repo] = {}

        if author not in commits_by_repo_dev[repo]:
            commits_by_repo_dev[repo][author] = 0

        commits_by_repo_dev[repo][author] += commits

    total_repos_with_commits = len(commits_by_repo_dev)
    logging.info("Aggregated commits for %d repositories", total_repos_with_commits)
    return commits_by_repo_dev


def compute_cursor_metrics(
    repos_with_min_stars: Set[str],
    cursor_commits_df: pd.DataFrame,
    cursor_developers: Dict[str, Set[str]],
    commits_by_repo_dev: Dict[str, Dict[str, int]],
    first_cursor_commit_times: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Compute cursor adoption metrics for each repository."""
    logging.info("Computing cursor adoption metrics...")
    results = {}

    for repo in repos_with_min_stars:
        # Count cursor file changing commits
        cursor_commits_count = len(
            cursor_commits_df[cursor_commits_df["repo_name"] == repo]
        )

        # Get cursor developers for this repo
        repo_cursor_devs = cursor_developers.get(repo, set())

        # Get all commits for this repo
        repo_commits = commits_by_repo_dev.get(repo, {})

        if not repo_commits:
            cursor_adoptor_percentage = 0.0
        else:
            # Count commits from cursor adoptor developers
            cursor_adoptor_commits = sum(
                commits
                for dev, commits in repo_commits.items()
                if dev in repo_cursor_devs
            )

            # Total commits in the repo
            total_commits = sum(repo_commits.values())

            # Calculate percentage
            cursor_adoptor_percentage = (
                (cursor_adoptor_commits / total_commits) * 100
                if total_commits > 0
                else 0.0
            )

        adoption_time = first_cursor_commit_times.get(repo, "")

        results[repo] = {
            "cursor_commits": cursor_commits_count,
            "cursor_adoptor_percentage": cursor_adoptor_percentage,
            "cursor_adoption_time": adoption_time,
        }

    repos_with_cursor_commits = sum(
        1 for r in results.values() if r["cursor_commits"] > 0
    )
    logging.info(
        "Computed metrics for %d repositories (%d with cursor commits)",
        len(results),
        repos_with_cursor_commits,
    )
    return results


def save_results(results: Dict[str, Dict[str, Any]], output_file: str) -> None:
    """Save results to CSV file."""
    logging.info("Saving results to %s", output_file)
    data = []
    for repo, metrics in results.items():
        data.append(
            {
                "repo_name": repo,
                "cursor_commits": metrics["cursor_commits"],
                "cursor_adoptor_percentage": metrics["cursor_adoptor_percentage"],
                "cursor_adoption_time": metrics["cursor_adoption_time"],
            }
        )

    results_df = pd.DataFrame(data)
    results_df = results_df.sort_values("cursor_adoptor_percentage", ascending=False)
    results_df.to_csv(output_file, index=False)

    logging.info("Results saved to %s", output_file)
    logging.info("Processed %d repositories", len(results))
    logging.info("Top 10 repositories by cursor adoptor percentage:")
    for _, row in results_df.head(10).iterrows():
        logging.info(
            "  %s: %.1f%% (%d commits)",
            row["repo_name"],
            row["cursor_adoptor_percentage"],
            row["cursor_commits"],
        )


def main() -> None:
    """Main function to compute repository-level cursor metrics."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info("Starting repository-level cursor metrics computation")

    try:
        repos_df, contributors_df, cursor_commits_df = load_data()

        # Filter repos with at least 10 stars
        repos_with_min_stars = filter_repos_by_stars(repos_df, min_stars=10)

        # Get cursor developers
        cursor_developers = get_cursor_developers(cursor_commits_df)

        # Count commits per developer
        commits_by_repo_dev = count_commits_per_developer(contributors_df)

        # Compute first cursor commit times per repo
        first_cursor_commit_times = compute_first_cursor_commit_times(cursor_commits_df)

        # Compute metrics
        results = compute_cursor_metrics(
            repos_with_min_stars,
            cursor_commits_df,
            cursor_developers,
            commits_by_repo_dev,
            first_cursor_commit_times,
        )

        # Save results
        save_results(results, "data/repo_metrics.csv")

        logging.info(
            "Repository-level cursor metrics computation completed successfully"
        )

    except Exception as e:
        logging.error("Failed to compute repository metrics: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
