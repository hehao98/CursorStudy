#!/usr/bin/env python3
"""
Script to prepare panel event data for cursor adoption analysis.

This script:
1. Reads repositories data from repos.csv
2. Reads time series data from ts_repos_{month/week}.csv
3. Creates a panel dataset with lead and lag indicators for cursor adoption events
4. Saves the results to panel_event_{weekly/monthly}.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
REPOS_CSV = DATA_DIR / "repos.csv"

# Lead and lag periods to generate
LEAD_PERIODS = range(1, 7)  # lead 1 to lead 6
LAG_PERIODS = range(0, 7)  # lag 0 to lag 6

# Fixed start date for data collection, matching run_sonarqube.py
START_DATE = "2024-01-01"
END_DATE = "2025-03-31"  # Drop anything beyond this to avoid incomplete data

DYNAMIC_METRICS = [
    "commits",
    "lines_added",
    "contributors",
    "stars",
    "issues",
    "issue_comments",
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

        padded_ts_dfs.append(merged_df)

    return pd.concat(padded_ts_dfs, ignore_index=True)


def generate_lead_lag_indicators(
    df: pd.DataFrame, time_col: str, event_time: str
) -> pd.DataFrame:
    """Generate lead and lag indicator variables for the event."""
    result_df = df.copy()

    # Convert times to datetime objects for comparison
    result_df["time_dt"] = pd.to_datetime(
        result_df[time_col] + ("-1" if time_col == "week" else "-01")
    )
    event_dt = pd.to_datetime(event_time + ("-1" if time_col == "week" else "-01"))

    # Calculate time to event (in periods)
    if time_col == "week":
        result_df["time_to_event"] = (
            ((result_df["time_dt"] - event_dt).dt.days / 7).round().astype(int)
        )
    else:  # month
        result_df["time_to_event"] = (
            result_df["time_dt"].dt.year - event_dt.year
        ) * 12 + (result_df["time_dt"].dt.month - event_dt.month)

    # Add event indicators and columns
    result_df["post_event"] = (result_df["time_dt"] >= event_dt).astype(int)
    result_df["event"] = event_time
    result_df["time"] = result_df[time_col]

    # Generate lead and lag indicators
    for lead in LEAD_PERIODS[:-1]:
        result_df[f"lead_{lead}"] = (result_df["time_to_event"] == -lead).astype(int)
    result_df[f"lead_{LEAD_PERIODS[-1]}"] = (
        result_df["time_to_event"] <= -LEAD_PERIODS[-1]
    ).astype(int)

    for lag in LAG_PERIODS[:-1]:
        result_df[f"lag_{lag}"] = (result_df["time_to_event"] == lag).astype(int)
    result_df[f"lag_{LAG_PERIODS[-1]}"] = (
        result_df["time_to_event"] >= LAG_PERIODS[-1]
    ).astype(int)

    return result_df.drop(columns=["time_dt"])


def prepare_panel_data(aggregation: str) -> pd.DataFrame:
    """Prepare panel data with lead and lag indicators for cursor adoption events."""
    # Set paths and time column
    ts_file = DATA_DIR / f"ts_repos_{aggregation}ly.csv"

    # Read repository data and time series data
    logging.info(f"Reading repository data from {REPOS_CSV}")
    repos_df = pd.read_csv(REPOS_CSV).dropna(subset=["repo_cursor_adoption"])

    # Convert adoption dates to the appropriate format
    date_format = "%Y-W%W" if aggregation == "week" else "%Y-%m"
    repos_df["adoption_time"] = repos_df["repo_cursor_adoption"].apply(
        lambda x: pd.to_datetime(x).strftime(date_format)
    )

    # Read time series data
    logging.info(f"Reading time series data from {ts_file}")
    ts_df = pd.read_csv(ts_file)

    # Pad missing time periods to ensure continuous time series data
    ts_df = pad_missing_periods(
        ts_df, group_columns=["repo_name"], time_col=aggregation
    )
    logging.info(f"Padded missing {aggregation}s in time series data")

    # Process each repository
    panel_dfs = []
    processed_repos = 0

    for _, repo in repos_df.iterrows():
        repo_name = repo["repo_name"]
        adoption_time = repo["adoption_time"]

        # Get time series data for this repository
        repo_ts = ts_df[ts_df["repo_name"] == repo_name].copy()

        if repo_ts.empty:
            logging.warning(f"No time series data found for repository: {repo_name}")
            continue

        # Generate lead and lag indicators
        repo_panel = generate_lead_lag_indicators(
            df=repo_ts, time_col=aggregation, event_time=adoption_time
        )

        panel_dfs.append(repo_panel)
        processed_repos += 1

    if not panel_dfs:
        logging.error("No repositories with sufficient data for panel analysis")
        return pd.DataFrame()

    # Combine all repository panel data
    panel_df = pd.concat(panel_dfs, ignore_index=True)
    logging.info(f"Processed {processed_repos} repositories for panel analysis")

    # Create a datetime column for filtering
    date_format = "%Y-W%W-1" if aggregation == "week" else "%Y-%m-01"
    panel_df["filter_date"] = pd.to_datetime(
        panel_df[aggregation] + ("-1" if aggregation == "week" else "-01")
    )
    start_date_dt = pd.to_datetime(START_DATE)
    end_date_dt = pd.to_datetime(END_DATE)

    # Filter data between START_DATE and END_DATE
    pre_filter_count = len(panel_df)

    # Filter for start date
    panel_df = panel_df[panel_df["filter_date"] >= start_date_dt]
    post_start_filter_count = len(panel_df)
    rows_filtered_start = pre_filter_count - post_start_filter_count

    # Filter for end date
    panel_df = panel_df[panel_df["filter_date"] <= end_date_dt]
    post_end_filter_count = len(panel_df)
    rows_filtered_end = post_start_filter_count - post_end_filter_count

    # Add repository age in days
    logging.info("Calculating repository age in days")
    repo_creation_dates = dict(
        zip(
            pd.read_csv(REPOS_CSV)["repo_name"],
            pd.to_datetime(pd.read_csv(REPOS_CSV)["repo_created"]).dt.tz_localize(None),
        )
    )
    panel_df["age"] = panel_df.apply(
        lambda row: (
            (row["filter_date"] - repo_creation_dates.get(row["repo_name"])).days
            if row["repo_name"] in repo_creation_dates
            else None
        ),
        axis=1,
    )

    # Drop the filtering column
    panel_df = panel_df.drop(columns=["filter_date"])

    logging.info(
        f"Filtered out {rows_filtered_start} rows prior to {START_DATE} and "
        f"{rows_filtered_end} rows after {END_DATE}"
    )

    # Reorder columns to put event-related columns right after time column
    event_columns = ["event", "post_event", "time_to_event"]

    # Add lead columns in order
    for lead in sorted(LEAD_PERIODS, reverse=True):
        event_columns.append(f"lead_{lead}")

    # Add lag columns in order
    for lag in sorted(LAG_PERIODS):
        event_columns.append(f"lag_{lag}")

    # Get all other columns that are not repo_name, time, or event-related
    other_columns = [
        col
        for col in panel_df.columns
        if col not in ["repo_name", aggregation, "time"] + event_columns
    ]

    # Reorder columns
    panel_df = panel_df[
        ["repo_name", aggregation, "time"] + event_columns + other_columns
    ]

    # Move age to come right after lines_added
    cols = panel_df.columns.tolist()
    if "age" in cols and "lines_added" in cols:
        lines_added_idx = cols.index("lines_added")
        cols.remove("age")
        cols.insert(lines_added_idx + 1, "age")
        panel_df = panel_df[cols]

    return panel_df


def main() -> None:
    """Main function to prepare panel event data."""
    parser = argparse.ArgumentParser(
        description="Prepare panel event data for cursor adoption analysis"
    )
    parser.add_argument(
        "--aggregation",
        choices=["week", "month"],
        default="week",
        help="Time aggregation format (week or month)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set output file and prepare panel data
    time_suffix = "weekly" if args.aggregation == "week" else "monthly"
    output_file = DATA_DIR / (f"panel_event_{time_suffix}.csv")

    logging.info(f"Preparing panel event data with {args.aggregation} aggregation")
    panel_df = prepare_panel_data(aggregation=args.aggregation)

    if panel_df.empty:
        logging.error("Failed to generate panel data")
        sys.exit(1)

    # Save results
    panel_df.drop(columns=[str(args.aggregation), "latest_commit"], inplace=True)
    panel_df.to_csv(output_file, index=False)
    logging.info(f"Saved panel event data with {len(panel_df)} rows to {output_file}")


if __name__ == "__main__":
    main()
