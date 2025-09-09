#!/usr/bin/env python3
"""
Script to prepare panel event data for cursor adoption analysis.

This script:
1. Reads repositories metadata from repos.csv
2. Reads time series data from ts_repos_{month/week}.csv and ts_repos_control_{month/week}.csv
3. Detects cursor adoption events from the cursor column in time series data
4. Creates a panel dataset with lead and lag indicators for cursor adoption events
5. Saves the results to panel_event_{weekly/monthly}.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
REPOS_CSV = DATA_DIR / "repos.csv"

WEEK_LEAD_AND_LAG = 30
MONTH_LEAD_AND_LAG = 6
# Fixed start date for data collection, matching run_sonarqube.py
START_DATE = "2024-01-01"
END_DATE = "2025-08-31"  # Drop anything beyond this to avoid incomplete data


def generate_lead_lag_indicators(
    df: pd.DataFrame,
    time_col: str,
    event_time: str,
    lead_periods=range(1, 7),
    lag_periods=range(0, 7),
) -> pd.DataFrame:
    """Generate lead and lag indicator variables for the event."""
    result_df = df.copy()

    # Convert times to datetime objects for comparison
    if time_col == "week":
        # Handle ISO week format properly
        result_df["time_dt"] = result_df[time_col].apply(
            lambda x: pd.to_datetime(x + "-1", format="%Y-W%W-%w")
        )
        event_dt = pd.to_datetime(event_time + "-1", format="%Y-W%W-%w")
    else:  # month
        result_df["time_dt"] = pd.to_datetime(result_df[time_col] + "-01")
        event_dt = pd.to_datetime(event_time + "-01")

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
    for lead in lead_periods[:-1]:
        result_df[f"lead_{lead}"] = (result_df["time_to_event"] == -lead).astype(int)
    result_df[f"lead_{lead_periods[-1]}"] = (
        result_df["time_to_event"] <= -lead_periods[-1]
    ).astype(int)

    for lag in lag_periods[:-1]:
        result_df[f"lag_{lag}"] = (result_df["time_to_event"] == lag).astype(int)
    result_df[f"lag_{lag_periods[-1]}"] = (
        result_df["time_to_event"] >= lag_periods[-1]
    ).astype(int)

    # Handle cursor abandonment: set post_event=0 and all indicators=0 when cursor=False
    cursor_abandonment_mask = (result_df["cursor"] == False) | (
        result_df["cursor"].isna()
    )

    # Set post_event to 0 for abandonment periods
    result_df.loc[cursor_abandonment_mask, "post_event"] = 0

    # Set all lag indicators to 0 for abandonment periods
    for lag in lag_periods:
        result_df.loc[cursor_abandonment_mask, f"lag_{lag}"] = 0

    return result_df.drop(columns=["time_dt"])


def process_non_adopter_repo_panel(
    repo_name: str,
    ts_df: pd.DataFrame,
    aggregation: str,
    lead_periods: range,
    lag_periods: range,
) -> pd.DataFrame:
    """Generate panel data for repositories that never adopted cursor."""
    # Get time series data for this repository
    repo_ts = ts_df[ts_df["repo_name"] == repo_name].copy()

    if repo_ts.empty:
        logging.warning(f"No time series data found for repository: {repo_name}")
        return None

    # Add required columns with appropriate null/zero values
    repo_ts["event"] = None
    repo_ts["post_event"] = 0
    repo_ts["time_to_event"] = None
    repo_ts["time"] = repo_ts[aggregation]

    # Add all lead and lag indicators as zeros
    for lead in lead_periods:
        repo_ts[f"lead_{lead}"] = 0

    for lag in lag_periods:
        repo_ts[f"lag_{lag}"] = 0

    return repo_ts


def detect_cursor_adoption_unified(
    combined_ts_df: pd.DataFrame, aggregation: str
) -> pd.DataFrame:
    """Detect cursor adoption events from unified time series data."""
    # Filter for rows where cursor=True
    cursor_adoptions = combined_ts_df[combined_ts_df["cursor"] == True].copy()

    if cursor_adoptions.empty:
        logging.warning("No cursor adoptions found in data")
        return pd.DataFrame(columns=["repo_name", "adoption_time"])

    # Find the first adoption time for each repository
    time_col = "week" if aggregation == "week" else "month"

    # Sort by time to ensure we get the earliest adoption
    cursor_adoptions = cursor_adoptions.sort_values(["repo_name", time_col])

    # Get first adoption for each repo
    first_adoptions = (
        cursor_adoptions.groupby("repo_name")[time_col].first().reset_index()
    )
    first_adoptions.rename(columns={time_col: "adoption_time"}, inplace=True)

    logging.info(f"Detected {len(first_adoptions)} repositories with cursor adoption")

    return first_adoptions


def load_data_unified(aggregation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine time series data with dynamic treatment assignment."""
    treatment_ts_file = DATA_DIR / f"ts_repos_{aggregation}ly.csv"
    control_ts_file = DATA_DIR / f"ts_repos_control_{aggregation}ly.csv"

    # Read both time series datasets
    logging.info(f"Reading treatment time series data from {treatment_ts_file}")
    treatment_ts_df = pd.read_csv(treatment_ts_file)

    logging.info(f"Reading control time series data from {control_ts_file}")
    control_ts_df = pd.read_csv(control_ts_file)

    # Add dataset source indicators for tracking
    treatment_ts_df["dataset_source"] = "treatment"
    control_ts_df["dataset_source"] = "control"

    # Combine datasets
    combined_ts_df = pd.concat([treatment_ts_df, control_ts_df], ignore_index=True)

    # Dynamic treatment assignment: is_treatment = cursor usage
    # This handles both adoption and discontinuation automatically
    # Each repo-period is treated based on actual cursor usage that period
    # Fill na values based on the last non-na value
    combined_ts_df["cursor"] = combined_ts_df["cursor"].fillna(method="ffill")
    combined_ts_df["is_treatment"] = combined_ts_df["cursor"].astype(int)

    # Detect cursor adoption events from unified data
    adoption_df = detect_cursor_adoption_unified(combined_ts_df, aggregation)

    logging.info(
        f"Combined {len(treatment_ts_df)} treatment and {len(control_ts_df)} "
        f"control observations into {len(combined_ts_df)} total observations"
    )

    return adoption_df, combined_ts_df


def process_repo_panel(
    repo_name: str,
    adoption_time: str,
    ts_df: pd.DataFrame,
    aggregation: str,
    lead_periods: range,
    lag_periods: range,
) -> pd.DataFrame:
    """Generate panel data for a single repository."""
    # Get time series data for this repository
    repo_ts = ts_df[ts_df["repo_name"] == repo_name].copy()

    if repo_ts.empty:
        logging.warning(f"No time series data found for repository: {repo_name}")
        return None

    # Generate lead and lag indicators
    return generate_lead_lag_indicators(
        df=repo_ts,
        time_col=aggregation,
        event_time=adoption_time,
        lead_periods=lead_periods,
        lag_periods=lag_periods,
    )


def filter_panel_data(panel_df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """Filter panel data by date range and add repository age."""
    # Create a datetime column for filtering
    if aggregation == "week":
        # Handle ISO week format properly
        panel_df["filter_date"] = panel_df[aggregation].apply(
            lambda x: pd.to_datetime(x + "-1", format="%Y-W%W-%w")
        )
    else:  # month
        panel_df["filter_date"] = pd.to_datetime(panel_df[aggregation] + "-01")

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

    # Drop the filtering column
    panel_df = panel_df.drop(columns=["filter_date"])

    logging.info(
        f"Filtered out {rows_filtered_start} rows prior to {START_DATE} and "
        f"{rows_filtered_end} rows after {END_DATE}"
    )

    return panel_df


def reorder_columns(
    panel_df: pd.DataFrame, aggregation: str, lead_periods: range, lag_periods: range
) -> pd.DataFrame:
    """Reorder columns in the panel dataframe for better organization."""
    # Reorder columns to put event-related columns right after time column
    event_columns = ["event", "post_event", "time_to_event"]

    # Add lead columns in order
    for lead in sorted(lead_periods, reverse=True):
        event_columns.append(f"lead_{lead}")

    # Add lag columns in order
    for lag in sorted(lag_periods):
        event_columns.append(f"lag_{lag}")

    # Get all other columns that are not repo_name, time, or event-related
    other_columns = [
        col
        for col in panel_df.columns
        if col not in ["repo_name", aggregation, "time", "is_treatment"] + event_columns
    ]

    # Reorder columns
    panel_df = panel_df[
        ["repo_name", aggregation, "time", "is_treatment"]
        + event_columns
        + other_columns
    ]

    return panel_df


def prepare_panel_data(aggregation: str) -> pd.DataFrame:
    """Prepare panel data with lead and lag indicators for cursor adoption events."""
    # Define lead and lag periods based on aggregation
    if aggregation == "week":
        lead_periods = range(1, WEEK_LEAD_AND_LAG + 1)
        lag_periods = range(0, WEEK_LEAD_AND_LAG + 1)
    else:
        lead_periods = range(1, MONTH_LEAD_AND_LAG + 1)
        lag_periods = range(0, MONTH_LEAD_AND_LAG + 1)

    # Load unified data with dynamic treatment assignment
    adoption_df, combined_ts_df = load_data_unified(aggregation)

    # Process repositories with cursor adoption events
    adopter_panel_dfs = []
    processed_adopter_repos = 0
    adopter_repos = set()

    for _, repo in adoption_df.iterrows():
        repo_panel = process_repo_panel(
            repo_name=repo["repo_name"],
            adoption_time=repo["adoption_time"],
            ts_df=combined_ts_df,
            aggregation=aggregation,
            lead_periods=lead_periods,
            lag_periods=lag_periods,
        )

        if repo_panel is not None:
            adopter_panel_dfs.append(repo_panel)
            processed_adopter_repos += 1
            adopter_repos.add(repo["repo_name"])

    # Process repositories that never adopted cursor
    non_adopter_panel_dfs = []
    processed_non_adopter_repos = 0

    for repo_name in combined_ts_df["repo_name"].unique():
        if repo_name in adopter_repos:
            continue
        repo_panel = process_non_adopter_repo_panel(
            repo_name=repo_name,
            ts_df=combined_ts_df,
            aggregation=aggregation,
            lead_periods=lead_periods,
            lag_periods=lag_periods,
        )

        if repo_panel is not None:
            non_adopter_panel_dfs.append(repo_panel)
            processed_non_adopter_repos += 1

    if not adopter_panel_dfs and not non_adopter_panel_dfs:
        logging.error("No repositories with sufficient data for panel analysis")
        return pd.DataFrame()

    # Combine all repository panel data
    panel_dfs = []
    if adopter_panel_dfs:
        panel_dfs.extend(adopter_panel_dfs)
    if non_adopter_panel_dfs:
        panel_dfs.extend(non_adopter_panel_dfs)

    panel_df = pd.concat(panel_dfs, ignore_index=True)
    logging.info(
        f"Processed {processed_adopter_repos} cursor adopter repositories and "
        f"{processed_non_adopter_repos} non-adopter repositories for panel analysis"
    )

    # Filter data and add repository age
    panel_df = filter_panel_data(panel_df, aggregation)

    # Reorder columns
    panel_df = reorder_columns(panel_df, aggregation, lead_periods, lag_periods)

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
