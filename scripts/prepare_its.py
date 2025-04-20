#!/usr/bin/env python3
"""
Script to prepare data for interrupted time series regression analysis.

This script:
1. Reads repositories data from repos.csv
2. Reads weekly time series data from ts_repos.csv and ts_contributors.csv
3. Filters repositories with sufficient history before Cursor adoption
4. Creates datasets suitable for interrupted time series analysis at both repository and contributor levels
5. Saves the results to its_repos.csv and its_contributors.csv
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Constants
REPOS_CSV = Path(__file__).parent.parent / "data" / "repos.csv"
TS_REPOS_CSV = Path(__file__).parent.parent / "data" / "ts_repos.csv"
TS_CONTRIBUTORS_CSV = Path(__file__).parent.parent / "data" / "ts_contributors.csv"
CURSOR_COMMITS_CSV = Path(__file__).parent.parent / "data" / "cursor_commits.csv"
REPO_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "its_repos.csv"
CONTRIBUTOR_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "its_contributors.csv"

DYNAMIC_METRICS = ["commits", "lines_added", "contributors"]
ACCUMULATIVE_METRICS = [
    "bugs",
    "vulnerabilities",
    "code_smells",
    "ncloc",
    "duplicated_lines_density",
    "comment_lines_density",
    "cognitive_complexity",
    "technical_debt",
]


def pad_missing_weeks(
    weekly_ts_df: pd.DataFrame, group_columns: List[str]
) -> pd.DataFrame:
    """
    Pad missing weeks in time series data.

    Args:
        weekly_ts_df: DataFrame containing weekly time series data
        group_columns: Columns to group by
    Returns:
        DataFrame with padded weeks for all entities
    """
    padded_ts_dfs = []
    for group_values, group_data in weekly_ts_df.groupby(group_columns):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        weeks = sorted(group_data["week"].unique())

        if len(weeks) <= 1:
            padded_ts_dfs.append(group_data)
            continue

        # Parse ISO week format strings to dates
        # Convert format like "2023-W01" to datetime objects
        start_date = pd.to_datetime(weeks[0] + "-1", format="%Y-W%W-%w")
        end_date = pd.to_datetime(weeks[-1] + "-1", format="%Y-W%W-%w")

        # Determine all weeks that should exist in the date range
        all_weeks = (
            pd.date_range(
                start=start_date,
                end=end_date,
                freq="W",
            )
            .strftime("%Y-W%W")
            .tolist()
        )

        # Create a dataframe with all weeks and the group values
        full_weeks_df = pd.DataFrame({"week": all_weeks})
        for i, col in enumerate(group_columns):
            full_weeks_df[col] = group_values[i]

        # Merge with existing data and fill missing values
        merged_df = pd.merge(
            full_weeks_df, group_data, on=group_columns + ["week"], how="left"
        )

        # Fill dynamic metrics with zeros
        for col in DYNAMIC_METRICS:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)

        # Forward-fill accumulative metrics from previous values
        for col in ACCUMULATIVE_METRICS:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].ffill()

        padded_ts_dfs.append(merged_df)

    return pd.concat(padded_ts_dfs, ignore_index=True)


def process_time_series(
    time_series_data: pd.DataFrame,
    entity_id: Dict[str, str],
    adoption_week: str,
    entity_type: str = "repository",
) -> List[Dict[str, Any]]:
    """
    Process time series data for interrupted time series analysis.

    Args:
        time_series_data: Weekly time series data for the entity
        entity_id: Dictionary with entity identification (repo_name, author_email, etc.)
        adoption_week: Week when adoption occurred
        entity_type: Type of entity ("repository" or "contributor")

    Returns:
        List of dictionaries with processed data for each week
    """
    # Extract entity name for logging
    if entity_type == "repository":
        entity_name = entity_id["repo_name"]
    else:
        entity_name = entity_id["author_name"]

    if time_series_data.empty:
        logging.debug(f"Skipping {entity_name}: No time series data available")
        return []

    weeks = np.sort(time_series_data["week"].unique())
    if len(weeks) == 0:
        logging.debug(f"Skipping {entity_name}: No week data available")
        return []

    # Find the adoption week index
    try:
        adoption_idx = np.where(weeks == adoption_week)[0][0]
    except IndexError:
        # If adoption week not in the data, find the closest later week
        future_weeks = [w for w in weeks if w > adoption_week]
        if not future_weeks:
            logging.debug(
                f"Skipping {entity_name}: No weeks after adoption week {adoption_week}"
            )
            return []
        original_adoption_week = adoption_week
        adoption_week = min(future_weeks)
        adoption_idx = np.where(weeks == adoption_week)[0][0]
        logging.debug(
            f"Using {adoption_week} as adoption week for {entity_name} (original: {original_adoption_week})"
        )

    logging.debug(f"Processing {entity_name}: Found adoption at week {adoption_week}")
    results = []

    # Process each week
    for week_idx, week in enumerate(weeks):
        week_data = time_series_data[time_series_data["week"] == week]
        if week_data.empty:
            continue

        # Calculate time and intervention variables
        intervention = 1 if week_idx >= adoption_idx else 0
        time_after_intervention = max(0, week_idx - adoption_idx)

        # Create base result with common fields
        result = {
            **entity_id,
            "week": week,
            "time": week_idx,
            "time_after_intervention": time_after_intervention,
            "intervention": intervention,
            "commits": int(week_data["commits"].iloc[0]),
            "lines_added": int(week_data["lines_added"].iloc[0]),
        }

        # Handle contributors column differently based on entity type
        if entity_type == "repository":
            result["contributors"] = int(week_data["contributors"].iloc[0])

            # Add additional metrics from ts_repos.csv if they exist
            for metric in ACCUMULATIVE_METRICS:
                if metric in week_data.columns and not pd.isna(
                    week_data[metric].iloc[0]
                ):
                    value = week_data[metric].iloc[0]
                    # Convert numeric values to int or float as appropriate
                    if isinstance(value, (int, float)):
                        try:
                            if value == int(value):
                                value = int(value)
                        except:
                            pass
                    result[metric] = value

        results.append(result)

    return results


def get_cursor_adopters(cursor_commits_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Get contributors who modified Cursor files and their first Cursor commit date.

    Args:
        cursor_commits_df: DataFrame containing cursor commits data

    Returns:
        Dictionary mapping contributors to their first cursor commit info
    """
    # Extract author name and email from the author field (format: "Name <email>")
    cursor_commits_df["author_email"] = cursor_commits_df["author"].str.extract(
        r"<([^>]+)>"
    )
    cursor_commits_df["author_name"] = cursor_commits_df["author"].apply(
        lambda x: x.split(" <")[0] if "<" in x else x
    )

    # Group by author_name and find the earliest cursor commit for each
    first_commits = {}

    for _, row in cursor_commits_df.sort_values("authored_at").iterrows():
        author_name = row["author_name"]
        if author_name not in first_commits:
            first_commits[author_name] = {
                "author_name": author_name,
                "author_email": row["author_email"],
                "repo_name": row["repo_name"],
                "authored_at": row["authored_at"],
                "cursor_file": row["cursor_file"],
                "adoption_week": datetime.fromisoformat(
                    row["authored_at"].replace("Z", "+00:00")
                ).strftime("%Y-W%W"),
            }

    logging.info("Found %d contributors who modified Cursor files", len(first_commits))
    return first_commits


def deduplicate_contributor_data(contributor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate and merge contributor data by author name.

    This handles cases where the same author might appear with different email addresses
    or other identifiers but should be treated as a single contributor.

    Args:
        contributor_df: DataFrame containing contributor time series data

    Returns:
        DataFrame with deduplicated contributor data
    """
    logging.info("Deduplicating contributor data by author name")

    # Group by author_name, repo_name, and week, then aggregate metrics
    deduplicated_df = (
        contributor_df.groupby(["repo_name", "author_name", "week"])
        .agg({"commits": "sum", "lines_added": "sum"})
        .reset_index()
    )

    # Count the number of duplicate entries that were merged
    total_entries = len(contributor_df)
    unique_entries = len(deduplicated_df)
    merged_entries = total_entries - unique_entries

    if merged_entries > 0:
        logging.info(
            f"Merged {merged_entries} duplicate entries ({total_entries} → {unique_entries})"
        )
    else:
        logging.info("No duplicate entries found")

    return deduplicated_df


def load_and_prepare_data(file_path: Path, group_columns: List[str]) -> pd.DataFrame:
    """
    Load data from a CSV file and pad missing weeks.

    Args:
        file_path: Path to the CSV file
        group_columns: Columns to group by when padding weeks

    Returns:
        Prepared DataFrame with padded weeks
    """
    logging.info(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)

    # If this is contributor data, deduplicate by author name first
    if "author" in df.columns:
        df["author_name"] = df["author"].apply(lambda x: x.split(" <")[0])
        df = deduplicate_contributor_data(df)

    logging.info(f"Padding missing weeks using {', '.join(group_columns)} as grouping")
    df = pad_missing_weeks(df, group_columns=group_columns)

    # Sort by the grouping columns and week
    sort_columns = group_columns + ["week"]
    df = df.sort_values(sort_columns)

    return df


def prepare_repo_its_dataset(weekly_ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare repository-level dataset for interrupted time series analysis.

    Args:
        weekly_ts_df: Weekly time series data

    Returns:
        DataFrame: Repository-level dataset for interrupted time series analysis
    """
    repos_df = pd.read_csv(REPOS_CSV)
    repos_df = repos_df.dropna(subset=["repo_cursor_adoption"])
    repos_df["adoption_week"] = repos_df["repo_cursor_adoption"].apply(
        lambda x: pd.to_datetime(x).strftime("%Y-W%W")
    )

    results = []
    valid_repos = 0

    logging.info("Processing repositories with cursor adoption dates")
    for _, repo in repos_df.iterrows():
        repo_name = repo["repo_name"]
        adoption_week = repo["adoption_week"]

        # Get time series data for this repository
        repo_data = weekly_ts_df[weekly_ts_df["repo_name"] == repo_name].copy()

        # Process the repository data
        repo_results = process_time_series(
            time_series_data=repo_data,
            entity_id={"repo_name": repo_name},
            adoption_week=adoption_week,
            # min_weeks_before_adoption=REPO_MIN_WEEKS_BEFORE_ADOPTION,
            entity_type="repository",
        )

        if repo_results:
            valid_repos += 1
            results.extend(repo_results)

    logging.info("Found %d repositories with sufficient data for analysis", valid_repos)
    return pd.DataFrame(results)


def prepare_contributor_its_dataset(contributor_ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare contributor-level dataset for interrupted time series analysis using contributor-specific time series data.

    Args:
        contributor_ts_df: Contributor time series data

    Returns:
        DataFrame: Contributor-level dataset for interrupted time series analysis
    """
    logging.info("Reading cursor commits data from %s", CURSOR_COMMITS_CSV)
    cursor_commits_df = pd.read_csv(CURSOR_COMMITS_CSV)

    # Get cursor adopters (contributors who modified Cursor files)
    cursor_adopters = get_cursor_adopters(cursor_commits_df)

    results = []
    valid_contributors = 0

    logging.info("Processing contributors with Cursor file modifications")
    for author_name, contributor_info in cursor_adopters.items():
        repo_name = contributor_info["repo_name"]
        adoption_week = contributor_info["adoption_week"]

        # Get time series data for this contributor and repository
        contributor_data = contributor_ts_df[
            (contributor_ts_df["author_name"] == author_name)
            & (contributor_ts_df["repo_name"] == repo_name)
        ].copy()

        # Process the contributor data
        contributor_results = process_time_series(
            time_series_data=contributor_data,
            entity_id={"repo_name": repo_name, "author_name": author_name},
            adoption_week=adoption_week,
            # min_weeks_before_adoption=CONTRIBUTOR_MIN_WEEKS_BEFORE_ADOPTION,
            entity_type="contributor",
        )

        if contributor_results:
            valid_contributors += 1
            results.extend(contributor_results)

    logging.info(
        "Found %d contributors with sufficient data for analysis", valid_contributors
    )
    return pd.DataFrame(results)


def main() -> None:
    """Main function to prepare regression data for both repository and contributor levels."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Prepare and save repository-level ITS data
    logging.info("Preparing repository-level ITS dataset")
    repo_ts_df = load_and_prepare_data(TS_REPOS_CSV, ["repo_name"])
    repo_results_df = prepare_repo_its_dataset(repo_ts_df)
    repo_results_df.to_csv(REPO_OUTPUT_FILE, index=False)
    logging.info(
        "Saved repository ITS dataset with %d rows to %s",
        len(repo_results_df),
        REPO_OUTPUT_FILE,
    )

    # Prepare and save contributor-level ITS data
    logging.info("Preparing contributor-level ITS dataset")
    contributor_results_df = prepare_contributor_its_dataset(
        load_and_prepare_data(TS_CONTRIBUTORS_CSV, ["author_name", "repo_name"])
    )
    contributor_results_df.to_csv(CONTRIBUTOR_OUTPUT_FILE, index=False)
    logging.info(
        "Saved contributor ITS dataset with %d rows to %s",
        len(contributor_results_df),
        CONTRIBUTOR_OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
