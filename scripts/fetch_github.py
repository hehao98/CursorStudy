#!/usr/bin/env python3
"""
Script to search GitHub for repositories with files matching .cursor patterns.

This script searches GitHub repositories for Cursor-related files using the GitHub
search API. It implements an adaptive partitioning strategy to overcome GitHub's
search result limit of 1000 items by:

1. First checking the total count of results for the search query
2. If results exceed 1000, automatically partitioning the search by file size
3. Using a binary-search-like approach to efficiently split the size ranges
4. Combining results from all partitions into a single dataset

The script handles rate limiting from GitHub's API and provides detailed logging
of the search process.
"""

import logging
import os
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from github import Github, RateLimitExceededException

# Load environment variables from .env file
load_dotenv()

# Settings for the search
QUERY = "path:.cursorrules OR path:.cursor/"
MAX_RETRIES = 5
RETRY_DELAY = 60
MAX_RESULTS_PER_PAGE = 100  # GitHub's maximum allowed results per page
MAX_SEARCH_RESULTS = 1000  # GitHub's maximum allowed search results
MAX_PARTITION_SIZE = 500  # Sometimes GitHub does not return accurate totalCount, so the partition must be smaller
TEMP_DIR = Path(__file__).parent.parent / "temp"  # Path to temp directory
DATA_DIR = Path(__file__).parent.parent / "data"  # Path to data directory


def ensure_dir(directory):
    """
    Ensure the directory exists, creating it if necessary.

    Args:
        directory (Path): Path to the directory

    Returns:
        Path: Path to the directory
    """
    if not directory.exists():
        directory.mkdir(parents=True)
        logging.info("Created directory at %s", directory)
    return directory


def ensure_temp_dir():
    """
    Ensure the temp directory exists, creating it if necessary.

    Returns:
        Path: Path to the temp directory
    """
    return ensure_dir(TEMP_DIR)


def ensure_data_dir():
    """
    Ensure the data directory exists, creating it if necessary.

    Returns:
        Path: Path to the data directory
    """
    return ensure_dir(DATA_DIR)


def get_github_token():
    """
    Retrieve GitHub token from environment variables.

    Returns:
        str: GitHub token

    Raises:
        ValueError: If no token is found
    """
    token = os.getenv("GITHUB_TOKEN")
    if token is None:
        # Try to load from .env in the project root if not in current directory
        root_env_path = Path(__file__).parent.parent / ".env"
        if root_env_path.exists():
            load_dotenv(dotenv_path=root_env_path)
            token = os.getenv("GITHUB_TOKEN")

    if token is None:
        raise ValueError(
            "GitHub token not provided. Add GITHUB_TOKEN to your .env file."
        )

    return token


def handle_rate_limit(func):
    """
    Decorator to handle GitHub API rate limiting.
    Retries the function when rate limit is exceeded.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with rate limit handling
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        for retry in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitExceededException:
                if retry < MAX_RETRIES - 1:
                    # Get current GitHub client from args or kwargs
                    g = None
                    for arg in args:
                        if isinstance(arg, Github):
                            g = arg
                            break

                    if g is None and "g" in kwargs:
                        g = kwargs["g"]

                    # Handle rate limit
                    if g:
                        rate_limit = g.get_rate_limit()
                        reset_time = rate_limit.search.reset.timestamp()
                        current_time = time.time()
                        sleep_time = max(reset_time - current_time + 5, RETRY_DELAY)
                    else:
                        sleep_time = RETRY_DELAY

                    logging.warning(
                        "Rate limit exceeded. Waiting for %.0f seconds...", sleep_time
                    )
                    time.sleep(sleep_time)
                else:
                    logging.warning("Max retries reached. Some results may be missing.")
                    return None
            except Exception as e:
                logging.error("Error in %s: %s", func.__name__, str(e))
                if retry < MAX_RETRIES - 1:
                    logging.info("Retrying in %d seconds...", RETRY_DELAY)
                    time.sleep(RETRY_DELAY)
                else:
                    logging.warning(
                        "Max retries reached. Returning partial or no results."
                    )
                    return None

    return wrapper


@handle_rate_limit
def get_search_count(query, token=None):
    """
    Get only the total count of results for a GitHub search query without fetching all results.

    Args:
        query (str): GitHub search query string
        token (str): GitHub personal access token

    Returns:
        int: Total count of search results
    """
    if token is None:
        token = get_github_token()

    g = Github(token)

    # Just get the search results without fetching pages
    search_results = g.search_code(query=query)
    total_count = search_results.totalCount
    logging.info("Query '%s' has %d total results", query, total_count)
    return total_count


@handle_rate_limit
def search_github_repos(query, token=None, output_file="github_cursor_repos.csv"):
    """
    Search GitHub for repositories containing files matching the specified query

    Args:
        query (str): GitHub search query string
        token (str): GitHub personal access token. If None, uses token from .env file.
        output_file (str): Path to save the results CSV file

    Returns:
        pandas.DataFrame: Results of the search
    """
    if token is None:
        token = get_github_token()

    # Ensure temp directory exists
    temp_dir = ensure_temp_dir()

    # Convert output_file to full path if it's not already
    if not os.path.isabs(output_file):
        output_file = temp_dir / output_file

    g = Github(token, per_page=MAX_RESULTS_PER_PAGE)
    logging.info(
        "Connected to GitHub. Rate limit: %d/%d",
        g.get_rate_limit().search.remaining,
        g.get_rate_limit().search.limit,
    )

    results = []
    page = 0
    has_more_results = True

    while has_more_results and page < (MAX_SEARCH_RESULTS // MAX_RESULTS_PER_PAGE + 1):
        logging.info(
            "Fetching page %d (up to %d results per page)...",
            page + 1,
            MAX_RESULTS_PER_PAGE,
        )

        # Get search results for this page
        search_results = g.search_code(query=query)

        # Log the total count of results
        if page == 0:
            logging.info(
                "Total search results available: %d", search_results.totalCount
            )

        # Process each result on this page
        current_page_results = list(search_results.get_page(page))
        for item in current_page_results:
            repo = item.repository

            result = {
                "repo_name": repo.full_name,
                "repo_url": repo.html_url,
                "file_path": item.path,
                "file_url": item.html_url,
                "repo_stars": repo.stargazers_count,
                "repo_forks": repo.forks_count,
                "repo_created": repo.created_at.isoformat(),
                "repo_updated": (
                    repo.updated_at.isoformat() if repo.updated_at else None
                ),
                "repo_description": repo.description,
            }

            results.append(result)
            logging.info("Found: %s - %s", repo.full_name, item.path)

        # Check if we have more pages
        if (
            len(current_page_results) < MAX_RESULTS_PER_PAGE
            or len(results) >= MAX_SEARCH_RESULTS
        ):
            has_more_results = False

        # Move to next page
        page += 1

    # Convert results to DataFrame
    df = pd.DataFrame(results) if results else pd.DataFrame()

    # Save results to CSV
    if not df.empty:
        df.to_csv(output_file, index=False)
        logging.info("Saved %d results to %s", len(df), output_file)
    else:
        logging.warning("No results found")

    return df


def search_with_efficient_partitioning(base_query, token=None, timestamp=None):
    """Search GitHub with efficient adaptive size partitioning using count checks first."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d")

    if token is None:
        token = get_github_token()

    # Ensure temp directory exists
    temp_dir = ensure_temp_dir()

    all_results = []
    combined_file = temp_dir / f"cursor_repos_{timestamp}_combined.csv"

    # Start with a queue of ranges to process
    # Each item is (min_size, max_size) where None represents unbounded
    ranges_to_process = [(None, None)]  # Start with unbounded search
    processed_ranges = []  # Track ranges we've already processed

    while ranges_to_process:
        min_size, max_size = ranges_to_process.pop(0)

        # Construct the size constraint for the query
        size_constraint = ""
        if min_size is not None and max_size is not None:
            size_constraint = f"size:{min_size}..{max_size}"
            size_label = f"size_{min_size}_to_{max_size}"
        elif min_size is not None:
            size_constraint = f"size:>={min_size}"
            size_label = f"size_gt_{min_size}"
        elif max_size is not None:
            size_constraint = f"size:<={max_size}"
            size_label = f"size_lt_{max_size}"
        else:
            size_constraint = ""  # No constraint
            size_label = "all_sizes"

        # Create the query with size constraint
        if size_constraint:
            query = f"{base_query} {size_constraint}"
        else:
            query = base_query

        logging.info("Checking count for query: %s", query)

        # FIRST: Get only the count without fetching results
        total_count = get_search_count(query, token=token)

        # Check if we need to partition this range
        if total_count >= MAX_PARTITION_SIZE:
            logging.warning(
                "Search count exceeds limit (%d > %d) with range [%s, %s]",
                total_count,
                MAX_PARTITION_SIZE,
                min_size or "MIN",
                max_size or "MAX",
            )

            # If we can't split further, fetch as many as we can
            if (
                min_size is not None
                and max_size is not None
                and max_size - min_size <= 1
            ):
                logging.warning(
                    "Cannot split range further. Will fetch partial results."
                )
                output_file = f"cursor_repos_{timestamp}_{size_label}.csv"
                df = search_github_repos(query, token=token, output_file=output_file)
                df["size_range"] = size_label
                all_results.append(df)
                processed_ranges.append((min_size, max_size))
            else:
                # Split the range and add to queue
                if min_size is None and max_size is None:
                    # First split: try a reasonable middle point like 10KB
                    ranges_to_process.append((None, 10))  # 0-10KB
                    ranges_to_process.append((10, None))  # >10KB
                elif min_size is None:
                    # Split based on max
                    middle = max_size // 2
                    ranges_to_process.append((None, middle))
                    ranges_to_process.append((middle, max_size))
                elif max_size is None:
                    # Split based on min and a reasonable increment
                    increment = min_size if min_size > 10 else 10
                    ranges_to_process.append((min_size, min_size + increment))
                    ranges_to_process.append((min_size + increment, None))
                else:
                    # Split the range in half
                    middle = (min_size + max_size) // 2
                    if middle > min_size:  # Ensure we can make progress
                        ranges_to_process.append((min_size, middle))
                        ranges_to_process.append((middle, max_size))
                    else:
                        # Can't split further
                        logging.warning(
                            "Cannot split range further. Will fetch partial results."
                        )
                        output_file = f"cursor_repos_{timestamp}_{size_label}.csv"
                        df = search_github_repos(
                            query, token=token, output_file=output_file
                        )
                        df["size_range"] = size_label
                        all_results.append(df)
                        processed_ranges.append((min_size, max_size))
        else:
            # This range has a reasonable number of results, fetch them
            logging.info(
                "Found %d results for range [%s, %s], fetching data...",
                total_count,
                min_size or "MIN",
                max_size or "MAX",
            )

            # Skip if no results or small number (like 0 or 1)
            if total_count <= 1:
                logging.info("Skipping range with ≤1 result")
                processed_ranges.append((min_size, max_size))
                continue

            output_file = f"cursor_repos_{timestamp}_{size_label}.csv"
            df = search_github_repos(query, token=token, output_file=output_file)
            df["size_range"] = size_label
            all_results.append(df)
            processed_ranges.append((min_size, max_size))

    # Combine all results if we have any
    if all_results:
        combined_df = pd.concat(all_results)
        combined_df = combined_df.drop_duplicates(subset=["repo_name", "file_path"])
        combined_df.to_csv(combined_file, index=False)
        logging.info(
            "Combined %d unique results from %d repositories saved to %s",
            len(combined_df),
            combined_df["repo_name"].nunique(),
            combined_file,
        )

        # Log statistics by size range
        logging.info("Results by size range:")
        size_stats = combined_df.groupby("size_range").size()
        for size_range, count in size_stats.items():
            logging.info("  - %s: %d files", size_range, count)

        return combined_df
    else:
        logging.warning("No results found across all partitions")
        return pd.DataFrame()


def process_results_to_data_csvs(input_file=None, timestamp=None):
    """
    Process the combined results from temp directory into repository and file CSVs in data directory.

    Args:
        input_file (str, optional): Path to the input CSV file. If None, uses the latest combined file.
        timestamp (str, optional): Timestamp to use for output filenames. If None, uses current date.

    Returns:
        tuple: (repos_df, files_df) DataFrames containing the processed data
    """
    # Ensure directories exist
    temp_dir = ensure_temp_dir()
    data_dir = ensure_data_dir()

    # Set timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d")

    # Find the input file if not specified
    if input_file is None:
        # Look for the latest combined file in temp directory
        combined_files = list(temp_dir.glob("cursor_repos_*_combined.csv"))
        if not combined_files:
            # Try any cursor_repos file
            combined_files = list(temp_dir.glob("cursor_repos_*.csv"))

        if not combined_files:
            raise FileNotFoundError(
                "No cursor repositories CSV files found in temp directory"
            )

        # Sort by modification time (most recent first)
        combined_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        input_file = combined_files[0]
    elif not os.path.isabs(input_file):
        input_file = temp_dir / input_file

    logging.info("Processing results from %s", input_file)

    # Read the combined results
    df = pd.read_csv(input_file)
    if df.empty:
        logging.warning("No data found in the input file")
        return pd.DataFrame(), pd.DataFrame()

    # Create a repositories DataFrame
    # Group by repository to get unique repositories
    repos_df = df[
        [
            "repo_name",
            "repo_stars",
            "repo_forks",
            "repo_created",
            "repo_updated",
            "repo_description",
        ]
    ].drop_duplicates(subset=["repo_name"])

    # Sort repositories by stars (descending)
    repos_df = repos_df.sort_values("repo_stars", ascending=False)

    # Create a files DataFrame
    files_df = df[["repo_name", "file_path", "file_url"]].copy()

    # Save to data directory
    repos_output = data_dir / "repos.csv"
    files_output = data_dir / "cursor_files.csv"

    repos_df.to_csv(repos_output, index=False)
    files_df.to_csv(files_output, index=False)

    logging.info("Saved %d unique repositories to %s", len(repos_df), repos_output)
    logging.info("Saved %d cursor files to %s", len(files_df), files_output)

    return repos_df, files_df


def main():
    """Main entry point of the script."""
    logging.basicConfig(
        format="%(asctime)s (PID %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Ensure the temp directory exists
    ensure_temp_dir()

    # Ensure the data directory exists
    ensure_data_dir()

    # Get GitHub token or prompt user to continue without it
    try:
        token = get_github_token()
    except ValueError:
        logging.warning(
            "No GitHub token found. The script may hit rate limits quickly."
        )
        logging.warning(
            "Add GITHUB_TOKEN=your_token to .env file to avoid rate limits."
        )
        proceed = input("Continue without token? (y/n): ").lower()
        if proceed != "y":
            logging.info("Exiting.")
            return
        token = None

    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d")

    # First check the total count to see if we need partitioning
    logging.info("Checking initial count for query: %s", QUERY)
    total_count = get_search_count(QUERY, token=token)

    result_df = None

    if total_count < MAX_PARTITION_SIZE:
        # If count is within limits, just do a regular search
        logging.info(
            "Query is within GitHub result limits. Proceeding with regular search."
        )
        output_file = f"cursor_repos_{timestamp}.csv"
        result_df = search_github_repos(QUERY, token=token, output_file=output_file)

        # Print summary
        if not result_df.empty:
            logging.info("\nSearch Summary:")
            logging.info(
                "Total repositories found: %d", result_df["repo_name"].nunique()
            )
            logging.info("Total files found: %d", len(result_df))
            logging.info("Top repositories by stars:")
            top_repos = (
                result_df.sort_values("repo_stars", ascending=False)
                .drop_duplicates("repo_name")
                .head(5)
            )
            for idx, row in top_repos.iterrows():
                logging.info("  - %s (%d stars)", row["repo_name"], row["repo_stars"])
    else:
        # If count exceeds limits, use adaptive partitioning
        logging.info(
            "Query exceeds GitHub result limits. Using adaptive size partitioning."
        )
        result_df = search_with_efficient_partitioning(
            QUERY, token=token, timestamp=timestamp
        )

    # Process results into data directory
    if result_df is not None and not result_df.empty:
        logging.info("Processing results into data directory...")
        process_results_to_data_csvs(timestamp=timestamp)


if __name__ == "__main__":
    main()
