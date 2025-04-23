#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from github import Github
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Global variables for file paths
REPOS_FILE = Path(__file__).parent.parent / "data" / "repos.csv"
EVENTS_FILE = Path(__file__).parent.parent / "data" / "repo_events.csv"
MATCHING_FILE = Path(__file__).parent.parent / "data" / "matching.csv"
CONTROL_REPOS_DIR = Path(__file__).parent.parent / "data"
EARLIEST_CONTROL_MONTH = "202408"
MAX_CONTROL_REPOS = 10000

# Global cache for repository languages
REPO_LANGUAGE_CACHE = {}


def get_github_token():
    """
    Retrieve GitHub token from environment variables.

    Returns:
        str: GitHub token

    Raises:
        ValueError: If no token is found
    """
    # Load environment variables from .env file
    load_dotenv(override=True)
    token = os.getenv("GITHUB_TOKEN")

    if token is None:
        raise ValueError(
            "GitHub token not provided. Add GITHUB_TOKEN to your .env file."
        )
    return token


def get_repository_primary_language(repo_name: str, github_client: Github):
    """
    Get the primary programming language for a repository.
    Uses a global cache to avoid duplicate API calls.

    Args:
        repo_name: Full name of the repository (owner/repo)
        github_client: GitHub API client

    Returns:
        str: Primary programming language or None if not found
    """
    global REPO_LANGUAGE_CACHE

    # Check if we already have this repository's language in the cache
    if repo_name in REPO_LANGUAGE_CACHE:
        return REPO_LANGUAGE_CACHE[repo_name]

    # Otherwise make the API call
    try:
        repo = github_client.get_repo(repo_name)
        language = repo.language
        # Store in cache
        REPO_LANGUAGE_CACHE[repo_name] = language
        return language

    except Exception as e:
        logging.warning(f"Error getting language for {repo_name}: {str(e)}")
        # Cache the failure as None
        REPO_LANGUAGE_CACHE[repo_name] = None

    return None


def load_cursor_adoption_repos(repos_path: Path) -> pd.DataFrame:
    """Load repositories that adopted Cursor from CSV file."""
    if not repos_path.exists():
        raise FileNotFoundError(f"Repos file not found: {repos_path}")

    repos_df = pd.read_csv(repos_path)
    repos_df = repos_df[repos_df["repo_cursor_adoption"].notna()]
    repos_df["repo_cursor_adoption"] = pd.to_datetime(repos_df["repo_cursor_adoption"])
    repos_df["adoption_month"] = repos_df["repo_cursor_adoption"].dt.strftime("%Y%m")
    logging.info("Loaded %d Cursor adoption repositories", len(repos_df))
    return repos_df


def load_repo_events(events_path: Path) -> pd.DataFrame:
    """Load repository events data from CSV file."""
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events_df = pd.read_csv(events_path)
    events_df["created_at"] = pd.to_datetime(events_df["created_at"]).dt.tz_localize(
        None
    )
    events_df["month"] = events_df["created_at"].dt.strftime("%Y%m")
    logging.info("Loaded events for %d repositories", events_df["repo"].nunique())
    return events_df


def load_control_repos(month: str) -> pd.DataFrame:
    """
    Load control repositories for a specific month.
    Collapses the data to one row per repository, keeping the latest 'within' period
    and summing all metrics across periods.
    """
    control_file = CONTROL_REPOS_DIR / f"control_repo_candidates_{month}.csv"
    control_df = pd.read_csv(control_file)
    metrics_columns = [
        "users_involved",
        "n_stars",
        "n_forks",
        "n_releases",
        "n_pulls",
        "n_issues",
        "n_comments",
        "total_events",
    ]

    # Sum metrics across all periods for each repository
    metrics_sum = control_df.groupby("repo_name")[metrics_columns].sum().reset_index()

    # Get the latest 'within' period for each repository
    within_data = control_df[control_df["period_type"] == "within"].copy()
    within_data["period"] = within_data["period"].astype(str)

    # Find the repository with the maximum period
    latest_period_idx = within_data.groupby("repo_name")["period"].idxmax()
    latest_data = within_data.loc[
        latest_period_idx, ["repo_name", "period", "age_days"]
    ]

    # Merge the summed metrics with the latest period data
    result_df = pd.merge(latest_data, metrics_sum, on="repo_name")

    # Add group identifier column
    result_df.insert(3, "group", "control")

    logging.info(
        "Loaded %d control repositories for month %s",
        len(result_df),
        month,
    )
    return result_df


def compute_repo_metrics(
    events_df: pd.DataFrame, repos_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute metrics for treatment repositories based on events before adoption.
    Returns one row per repository with summed metrics from all events prior to cursor adoption.
    Uses groupby operations for better performance.
    """
    logging.info("Computing metrics for treatment repositories")

    # Create a mapping from repo name to adoption month
    repo_adoption = repos_df.set_index("repo_name")["adoption_month"].to_dict()

    # Filter only events for repositories in our adoption list
    valid_repos = set(repo_adoption.keys())
    filtered_events = events_df[events_df["repo"].isin(valid_repos)].copy()

    if filtered_events.empty:
        logging.warning("No events found for treatment repositories")
        return pd.DataFrame()

    # Add adoption month to each event
    filtered_events["adoption_month"] = filtered_events["repo"].map(repo_adoption)

    # Filter events that occurred before adoption
    pre_adoption_events = filtered_events[
        filtered_events["month"] < filtered_events["adoption_month"]
    ]

    if pre_adoption_events.empty:
        logging.warning("No pre-adoption events found for treatment repositories")
        return pd.DataFrame()

    # Get first event date for each repository
    first_events = filtered_events.groupby("repo")["created_at"].min()

    # Get the latest month before adoption for each repository
    latest_months = pre_adoption_events.groupby("repo")["month"].max().reset_index()
    latest_months.rename(columns={"month": "period"}, inplace=True)

    # Calculate metrics for each repository using aggregation
    metrics = (
        pre_adoption_events.groupby("repo")
        .agg(
            users_involved=("actor", "nunique"),
            n_stars=("type", lambda x: (x == "WatchEvent").sum()),
            n_forks=("type", lambda x: (x == "ForkEvent").sum()),
            n_releases=("type", lambda x: (x == "ReleaseEvent").sum()),
            n_pulls=("type", lambda x: (x == "PullRequestEvent").sum()),
            n_issues=("type", lambda x: (x == "IssuesEvent").sum()),
            n_comments=(
                "type",
                lambda x: x.isin(
                    ["IssueCommentEvent", "PullRequestReviewCommentEvent"]
                ).sum(),
            ),
            total_events=("type", "count"),
        )
        .reset_index()
    )

    # Merge metrics with latest month
    result_df = pd.merge(metrics, latest_months, on="repo")

    # Calculate age_days
    result_df["month_end"] = (
        pd.to_datetime(result_df["period"] + "01").dt.tz_localize(None)
        + pd.DateOffset(months=1)
        - pd.Timedelta(days=1)
    )

    # Add first event date and calculate age_days
    result_df = result_df.merge(
        first_events.reset_index().rename(columns={"created_at": "first_event"}),
        on="repo",
    )

    result_df["age_days"] = (result_df["month_end"] - result_df["first_event"]).dt.days
    result_df["age_days"] = result_df["age_days"].clip(lower=0)

    # Clean up and rename columns
    result_df = result_df.drop(columns=["month_end", "first_event"])
    result_df.rename(columns={"repo": "repo_name"}, inplace=True)

    # Add group identifier
    result_df.insert(3, "group", "treatment")

    logging.info(
        "Computed metrics for %d treatment repositories",
        len(result_df),
    )
    return result_df


def compute_propensity_scores(
    treatment_df: pd.DataFrame, control_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute propensity scores for treatment and control groups using simplified features.
    Uses direct metrics without time-series transformations.
    """
    logging.info("Computing propensity scores")

    # Create a unified dataset for modeling
    treatment_df["treatment"] = 1
    control_df["treatment"] = 0
    combined_df = pd.concat([treatment_df, control_df], ignore_index=True)

    # Define features for propensity score model
    feature_list = [
        "age_days",
        "users_involved",
        "n_stars",
        "n_forks",
        "n_releases",
        "n_pulls",
        "n_issues",
        "n_comments",
        "total_events",
    ]

    # Check which features are actually available in the data
    available_features = [col for col in feature_list if col in combined_df.columns]

    # Prepare feature matrix and target vector
    X = combined_df[available_features]
    y = combined_df["treatment"]

    # Log which features are being used
    logging.info("Using features for propensity score model: %s", available_features)

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    # Calculate precision and recall
    y_pred = model.predict(X_scaled)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    logging.info("Precision: %.4f, Recall: %.4f", precision, recall)

    # Calculate McFadden's pseudo R-squared
    # First, get the log-likelihood of the full model
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    ll_full = sum(
        y * np.log(y_pred_proba + 1e-10) + (1 - y) * np.log(1 - y_pred_proba + 1e-10)
    )
    # Then, calculate the log-likelihood of the null model (intercept only)
    null_prob = sum(y) / len(y)
    ll_null = sum(
        y * np.log(null_prob + 1e-10) + (1 - y) * np.log(1 - null_prob + 1e-10)
    )
    # Calculate McFadden's pseudo R-squared
    mcfadden_r2 = 1 - (ll_full / ll_null)
    logging.info("McFadden's pseudo R-squared: %.4f", mcfadden_r2)

    # Calculate propensity scores using normalized features
    propensity_scores = model.predict_proba(X_scaled)[:, 1]
    combined_df["propensity_score"] = propensity_scores

    logging.info("Computed propensity scores for %d repositories", len(combined_df))

    return combined_df


def perform_nearest_neighbor_matching(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform nearest neighbor matching for each treatment repository within matched periods.
    Finds up to three control repositories with the closest propensity scores and same primary language.

    Args:
        summary_df: DataFrame with one row per repository including propensity scores

    Returns:
        DataFrame with matched_control columns added for treatment repositories
    """
    # Initialize the matched columns for up to three matches
    for i in range(1, 4):
        summary_df[f"matched_control_{i}"] = ""
        summary_df[f"matched_language_{i}"] = ""
        summary_df[f"propensity_score_diff_{i}"] = None

    # Load repositories data to get primary languages for treatment repositories
    repos_df = pd.read_csv(REPOS_FILE)

    # Try to get GitHub token for API access
    try:
        token = get_github_token()
        github_client = Github(token)
        logging.info(
            "Connected to GitHub for language matching. Rate limit: %d/%d",
            github_client.get_rate_limit().core.remaining,
            github_client.get_rate_limit().core.limit,
        )
    except ValueError:
        logging.warning("No GitHub token found. Language matching will be skipped.")
        github_client = None

    # Process each matched period separately
    for period in summary_df["matched_period"].unique():
        period_df = summary_df[summary_df["matched_period"] == period]

        treatment = period_df[period_df["group"] == "treatment"]
        control = period_df[period_df["group"] == "control"].copy()

        # For each treatment repository, find closest control repositories with same language
        for _, treat_row in treatment.iterrows():
            repo_name = treat_row["repo_name"]
            treat_score = treat_row["propensity_score"]
            treat_language = repos_df[repos_df["repo_name"] == repo_name][
                "repo_primary_language"
            ].iloc[0]
            if pd.isna(treat_language) or treat_language == "":
                treat_language = None

            logging.info(
                "Checking treatment %s (score: %.4f, language: %s)",
                repo_name,
                treat_score,
                treat_language,
            )

            control["score_diff"] = (
                control["propensity_score"] - treat_row["propensity_score"]
            ).abs()

            sorted_control = control.sort_values("score_diff")

            # Keep track of matched controls to avoid duplicates
            matched_controls = set()
            match_count = 0

            for _, control_row in sorted_control.iterrows():
                if match_count >= 3:  # Stop after finding 3 matches
                    break

                control_name = control_row["repo_name"]

                # Skip if this control is already matched
                if control_name in matched_controls:
                    continue

                control_score = control_row["propensity_score"]
                score_diff = control_row["score_diff"]

                # control_language = get_repository_primary_language(
                #    control_name, github_client
                # )
                control_language = treat_language  # turns off language matching for now

                if treat_language is None or (control_language == treat_language):
                    match_count += 1
                    matched_controls.add(control_name)

                    # Update the appropriate columns for this match
                    summary_df.loc[
                        (summary_df["repo_name"] == repo_name)
                        & (summary_df["matched_period"] == period),
                        f"matched_control_{match_count}",
                    ] = control_name

                    logging.info(
                        "Matched %s (score: %.4f) with %s (score: %.4f), diff: %.4f, language: %s, match #%d",
                        repo_name,
                        treat_score,
                        control_name,
                        control_score,
                        score_diff,
                        control_language,
                        match_count,
                    )

    return summary_df


def create_matching_summary(propensity_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Create a summary dataframe with one row per repository including aggregated metrics.

    Args:
        propensity_dfs: List of dataframes with propensity scores and metrics

    Returns:
        Summary dataframe with one row per repository
    """
    if not propensity_dfs:
        return pd.DataFrame()

    # Combine all propensity data
    combined_propensity_df = pd.concat(propensity_dfs, ignore_index=True)

    # Get the latest period for each repo
    latest_periods = (
        combined_propensity_df[combined_propensity_df["period_type"] == "within"]
        .groupby("repo_name")["period"]
        .max()
        .reset_index()
    )

    # Get metrics to sum
    metrics_columns = [
        "users_involved",
        "n_stars",
        "n_forks",
        "n_releases",
        "n_pulls",
        "n_issues",
        "n_comments",
        "total_events",
    ]

    # Aggregate to one row per repository
    summary_df = (
        combined_propensity_df.groupby("repo_name")
        .agg(
            {
                "group": "first",
                "propensity_score": "first",
                "age_days": "max",
                **{col: "sum" for col in metrics_columns},
            }
        )
        .reset_index()
    )

    # Add the matched_period (latest period)
    summary_df = summary_df.merge(latest_periods, on="repo_name", how="left")
    summary_df = summary_df.rename(columns={"period": "matched_period"})

    # Reorder columns
    column_order = [
        "repo_name",
        "matched_period",
        "group",
        "propensity_score",
        "age_days",
    ] + metrics_columns
    summary_df = summary_df[column_order]

    # Sort the dataframe
    summary_df = summary_df.sort_values(by=["matched_period", "group", "repo_name"])

    # Perform nearest neighbor matching
    summary_df = perform_nearest_neighbor_matching(summary_df)

    return summary_df


def main() -> None:
    """Main entry point for repository matching script."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load treatment repositories and events data
    cursor_repos = load_cursor_adoption_repos(REPOS_FILE)
    events_df = load_repo_events(EVENTS_FILE)

    # Filter to months from EARLIEST_CONTROL_MONTH onwards
    valid_months = [
        m
        for m in cursor_repos["adoption_month"].unique()
        if m >= EARLIEST_CONTROL_MONTH
    ]

    # Compute metrics for all treatment repositories at once
    treatment_df = compute_repo_metrics(
        events_df, cursor_repos[cursor_repos["adoption_month"].isin(valid_months)]
    )

    all_propensity_data = []

    for month in sorted(valid_months):
        logging.info("Loading control repositories for month: %s", month)
        treat_repos = cursor_repos[cursor_repos["adoption_month"] == month]
        treat = treatment_df[treatment_df["repo_name"].isin(treat_repos["repo_name"])]
        control = load_control_repos(month)
        control = control[~control["repo_name"].isin(treat_repos["repo_name"])]
        control = control[
            control["repo_name"].isin(control["repo_name"].sample(n=MAX_CONTROL_REPOS))
        ]

        propensity_df = compute_propensity_scores(treat.copy(), control)
        all_propensity_data.append(propensity_df)

    # Create and save the summary dataframe
    summary_df = create_matching_summary(all_propensity_data)
    if not summary_df.empty:
        summary_df.to_csv(MATCHING_FILE, index=False)
        logging.info(
            f"Saved aggregated matching data to {MATCHING_FILE} with {len(summary_df)} repositories"
        )

    logging.info("Repository metrics computation complete.")


if __name__ == "__main__":
    main()
