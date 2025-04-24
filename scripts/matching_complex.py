#!/usr/bin/env python3
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from github import Github
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Global variables for file paths
REPOS_FILE = Path(__file__).parent.parent / "data" / "repos.csv"
EVENTS_FILE = Path(__file__).parent.parent / "data" / "repo_events.csv"
MATCHING_FILE = Path(__file__).parent.parent / "data" / "matching.csv"
CONTROL_REPOS_DIR = Path(__file__).parent.parent / "data"
EARLIEST_CONTROL_MONTH = "202408"
MAX_CONTROL_REPOS = 10000

# Model configuration
USE_RANDOM_FOREST = False  # Deprecated, Random Forest is not used anymore
USE_LANGUAGE_MATCHING = True  # True to match repositories with the same language

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
    """Load control repositories for a specific month."""
    control_file = CONTROL_REPOS_DIR / f"control_repo_candidates_{month}.csv"

    if not control_file.exists():
        logging.warning(
            "Control repo file not found for month %s: %s", month, control_file
        )
        return pd.DataFrame()

    # Load control repo data - already contains the metrics
    control_df = pd.read_csv(control_file)

    # Add group identifier column
    control_df["group"] = "control"

    logging.info(
        "Loaded %d control repositories for month %s",
        control_df["repo_name"].nunique(),
        month,
    )
    return control_df


def compute_repo_metrics(
    events_df: pd.DataFrame, repos_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute time series metrics for treatment repositories based on events before adoption."""
    logging.info("Computing metrics for treatment repositories")

    results = []

    # Process each repository
    for repo_name, repo_events in events_df.groupby("repo"):
        # Skip repos not in our list
        if repo_name not in repos_df["repo_name"].values:
            continue

        # Get adoption month and first event date
        adoption_month = repos_df.loc[
            repos_df["repo_name"] == repo_name, "adoption_month"
        ].iloc[0]
        repo_first_event = repo_events["created_at"].min()

        # Generate periods: 6 months prior to adoption + sum for older events
        adoption_dt = pd.to_datetime(adoption_month + "01").tz_localize(None)
        periods = [
            (adoption_dt - pd.DateOffset(months=i)).strftime("%Y%m")
            for i in range(1, 8)
        ]
        period_types = ["within"] * 6 + ["sum"]

        # Process each period
        for i, (period, period_type) in enumerate(zip(periods, period_types)):
            # Filter events for this period
            if period_type == "within":
                period_events = repo_events[repo_events["month"] == period]
                month_end = (
                    pd.to_datetime(period + "01").tz_localize(None)
                    + pd.DateOffset(months=1)
                    - pd.Timedelta(days=1)
                )
            else:
                period_events = repo_events[repo_events["month"] < periods[5]]
                month_end = pd.to_datetime(periods[5] + "01").tz_localize(
                    None
                ) - pd.Timedelta(days=1)

            # Add metrics to results
            results.append(
                {
                    "repo_name": repo_name,
                    "period": period,
                    "period_type": period_type,
                    "group": "treatment",
                    "age_days": max(0, (month_end - repo_first_event).days),
                    "users_involved": (
                        period_events["actor"].nunique()
                        if not period_events.empty
                        else 0
                    ),
                    "n_stars": period_events["type"].eq("WatchEvent").sum(),
                    "n_forks": period_events["type"].eq("ForkEvent").sum(),
                    "n_releases": period_events["type"].eq("ReleaseEvent").sum(),
                    "n_pulls": period_events["type"].eq("PullRequestEvent").sum(),
                    "n_issues": period_events["type"].eq("IssuesEvent").sum(),
                    "n_comments": period_events["type"]
                    .isin(["IssueCommentEvent", "PullRequestReviewCommentEvent"])
                    .sum(),
                    "total_events": len(period_events),
                }
            )

    metrics_df = pd.DataFrame(results)
    logging.info(
        "Computed metrics for %d treatment repositories",
        metrics_df["repo_name"].nunique(),
    )
    return metrics_df


def compute_propensity_scores(
    treatment_df: pd.DataFrame, control_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute propensity scores for treatment and control groups."""
    logging.info("Computing propensity scores")

    # Create a unified dataset for modeling
    treatment_df["treatment"] = 1
    control_df["treatment"] = 0
    combined_df = pd.concat([treatment_df, control_df], ignore_index=True)

    # Get the most recent period for each repo (for age_days)
    combined_df["period"] = combined_df["period"].astype(str)
    latest_period_df = (
        combined_df[combined_df["period_type"] == "within"]
        .groupby("repo_name")["period"]
        .max()
        .reset_index()
    )
    latest_age_df = combined_df.merge(latest_period_df, on=["repo_name", "period"])[
        ["repo_name", "age_days"]
    ].drop_duplicates()

    # Create features by pivoting metrics across time periods
    feature_list = [
        "users_involved",
        "n_stars",
        "n_forks",
        "n_releases",
        "n_pulls",
        "n_issues",
        "n_comments",
        "total_events",
    ]

    # Initialize feature dataframe with age_days
    features_df = latest_age_df.copy()

    # Add features for each time period (6 monthly periods + sum)
    for metric in feature_list:
        # Pivot to create columns for each period
        # Add monthly metrics (period_type = 'within')
        monthly_pivot = combined_df[combined_df["period_type"] == "within"].pivot(
            index="repo_name", columns="period", values=metric
        )

        # Rename columns to indicate metric and period
        monthly_pivot.columns = [f"{metric}_{col}" for col in monthly_pivot.columns]

        # Add total metrics (period_type = 'sum')
        sum_metrics = combined_df[combined_df["period_type"] == "sum"].set_index(
            "repo_name"
        )[metric]
        sum_metrics.name = f"{metric}_sum"

        # Add to features dataframe
        features_df = features_df.merge(
            monthly_pivot, left_on="repo_name", right_index=True, how="left"
        )
        features_df = features_df.merge(
            sum_metrics, left_on="repo_name", right_index=True, how="left"
        )

    # Fill NAs (repositories without data for certain periods)
    features_df = features_df.fillna(0)

    # Add treatment label
    features_df = features_df.merge(
        combined_df[["repo_name", "treatment"]].drop_duplicates(),
        on="repo_name",
        how="left",
    )

    logging.info("Training model")
    X = features_df.drop(["repo_name", "treatment"], axis=1)
    y = features_df["treatment"]

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select model based on global configuration
    if USE_RANDOM_FOREST:
        model = RandomForestClassifier(random_state=42)
        logging.info("Training Random Forest model")
    else:
        model = LogisticRegression(random_state=42)
        logging.info("Training Logistic Regression model")

    model.fit(X_scaled, y)

    # Calculate AUC score
    auc_score = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])
    logging.info("AUC score: %.4f", auc_score)

    # Calculate McFadden's pseudo R-squared (for logistic regression only)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    if not USE_RANDOM_FOREST:
        # First, get the log-likelihood of the full model
        ll_full = sum(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
        # Then, calculate the log-likelihood of the null model (intercept only)
        null_prob = sum(y) / len(y)
        ll_null = sum(y * np.log(null_prob) + (1 - y) * np.log(1 - null_prob))
        # Calculate McFadden's pseudo R-squared
        mcfadden_r2 = 1 - (ll_full / ll_null)
        logging.info("McFadden's pseudo R-squared: %.4f", mcfadden_r2)

    # Calculate propensity scores using normalized features
    propensity_scores = model.predict_proba(X_scaled)[:, 1]
    features_df["propensity_score"] = propensity_scores

    logging.info("Computed propensity scores for %d repositories", len(features_df))

    # Add propensity scores back to the original dataframes
    result_df = combined_df.merge(
        features_df[["repo_name", "propensity_score"]], on="repo_name", how="left"
    )

    return result_df


def perform_nearest_neighbor_matching(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform nearest neighbor matching for each treatment repository within matched periods.
    Finds up to three control repositories with the closest propensity scores.

    If USE_LANGUAGE_MATCHING is True, also requires the same primary language for matching.

    Args:
        summary_df: DataFrame with one row per repository including propensity scores

    Returns:
        DataFrame with matched_control columns added for treatment repositories
    """
    # Initialize the matched columns for up to three matches
    for i in range(1, 4):
        summary_df[f"matched_control_{i}"] = ""

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

                # Only check language if language matching is enabled
                if USE_LANGUAGE_MATCHING:
                    control_language = get_repository_primary_language(
                        control_name, github_client
                    )
                    language_match = treat_language is None or (
                        control_language == treat_language
                    )
                else:
                    control_language = None
                    language_match = True

                if language_match:
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
                        "ignored" if not USE_LANGUAGE_MATCHING else control_language,
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
