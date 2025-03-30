#!/usr/bin/env python3
"""
Script to search GitHub for repositories with files matching the pattern '^\.cursor'.
This will find all files starting with .cursor in GitHub repositories.
"""

import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from github import Github, RateLimitExceededException

# Load environment variables from .env file
load_dotenv()

QUERY = "path:.cursorrules OR path:.cursor/"

# GitHub has a search rate limit, so we need to handle that
MAX_RETRIES = 5
RETRY_DELAY = 60  # seconds
MAX_RESULTS_PER_PAGE = 100  # GitHub's maximum allowed results per page


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
    # Initialize GitHub connection
    if token is None:
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

    g = Github(token, per_page=MAX_RESULTS_PER_PAGE)
    logging.info(
        "Connected to GitHub. Rate limit: %d/%d",
        g.get_rate_limit().search.remaining,
        g.get_rate_limit().search.limit,
    )

    results = []
    page = 0
    has_more_results = True

    while has_more_results:
        for retry in range(MAX_RETRIES):
            try:
                logging.info(
                    "Fetching page %d (up to %d results per page)...",
                    page + 1,
                    MAX_RESULTS_PER_PAGE,
                )

                # Get search results for this page
                search_results = g.search_code(query=query)

                # Log the total count of results
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
                        "repo_updated": repo.updated_at.isoformat()
                        if repo.updated_at
                        else None,
                        "repo_description": repo.description,
                    }

                    results.append(result)
                    logging.info("Found: %s - %s", repo.full_name, item.path)

                # Check if we have more pages
                if (
                    len(current_page_results) < MAX_RESULTS_PER_PAGE
                ):  # If we got fewer results than max, we're on the last page
                    has_more_results = False

                # Move to next page
                page += 1

                # Break out of retry loop if successful
                break

            except RateLimitExceededException:
                rate_limit = g.get_rate_limit()
                reset_time = rate_limit.search.reset.timestamp()
                current_time = time.time()
                sleep_time = max(reset_time - current_time + 5, RETRY_DELAY)

                logging.warning(
                    "Rate limit exceeded. Waiting for %.0f seconds...", sleep_time
                )
                time.sleep(sleep_time)

                if retry == MAX_RETRIES - 1:
                    logging.warning("Max retries reached. Saving current results.")
                    has_more_results = False

            except Exception as e:
                logging.error("Error: %s", str(e))
                if retry == MAX_RETRIES - 1:
                    logging.warning("Max retries reached. Saving current results.")
                    has_more_results = False
                else:
                    logging.info("Retrying in %d seconds...", RETRY_DELAY)
                    time.sleep(RETRY_DELAY)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Save results to CSV
    if not df.empty:
        df.to_csv(output_file, index=False)
        logging.info("Saved %d results to %s", len(df), output_file)
    else:
        logging.warning("No results found")

    return df


def main():
    logging.basicConfig(
        format="%(asctime)s (PID %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Check for token
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logging.warning("No GITHUB_TOKEN found in .env file.")
        logging.warning(
            "This script may hit rate limits quickly without authentication."
        )
        logging.warning(
            "It's recommended to create a .env file with your GitHub token:"
        )
        logging.warning("GITHUB_TOKEN=your_token_here")
        proceed = input("Continue without token? (y/n): ").lower()
        if proceed != "y":
            logging.info("Exiting.")
            return

    # Run the search
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file = f"cursor_repos_{timestamp}.csv"

    logging.info("Searching GitHub repositories...")
    df = search_github_repos(token=token, output_file=output_file, query=QUERY)

    # Print summary
    if not df.empty:
        logging.info("\nSearch Summary:")
        logging.info("Total repositories found: %d", df["repo_name"].nunique())
        logging.info("Total files found: %d", len(df))
        logging.info("Top repositories by stars:")
        top_repos = (
            df.sort_values("repo_stars", ascending=False)
            .drop_duplicates("repo_name")
            .head(5)
        )
        for idx, row in top_repos.iterrows():
            logging.info("  - %s (%d stars)", row["repo_name"], row["repo_stars"])


if __name__ == "__main__":
    main()
