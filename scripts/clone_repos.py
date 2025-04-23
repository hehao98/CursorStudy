#!/usr/bin/env python3
"""
Script to clone repositories with 10 or more stars from the repos.csv file.

This script reads the repos.csv file and clones repositories that have 10 or more
stars into the ../CursorRepos folder. It implements proper logging and error handling
for the cloning process.
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Constants
REPOS_CSV = Path(__file__).parent.parent / "data" / "repos.csv"
CONTROL_REPOS_CSV = Path(__file__).parent.parent / "data" / "matching.csv"
CLONE_DIR = Path(__file__).parent.parent.parent / "CursorRepos"
CONTROL_CLONE_DIR = Path(__file__).parent.parent.parent / "ControlRepos"
MIN_STARS = 10


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


def is_git_repo(path):
    """
    Check if the given path is a valid Git repository.

    Args:
        path (Path): Path to check

    Returns:
        bool: True if it's a valid Git repository, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == "true"
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


def pull_latest_changes(repo_path, repo_name):
    """
    Pull the latest changes for an existing repository.

    Args:
        repo_path (Path): Path to the repository
        repo_name (str): Name of the repository

    Returns:
        bool: True if pull was successful, False otherwise
    """
    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "pull", "--ff-only"],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.info("Successfully updated %s", repo_name)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Failed to update %s: %s", repo_name, e.stderr.strip())
        return False
    except Exception as e:
        logging.error("Failed to update %s: %s", repo_name, str(e))
        return False


def clone_repository(repo_name, clone_path):
    """
    Clone a repository using git.

    Args:
        repo_name (str): Name of the repository in format "owner/repo"
        clone_path (Path): Path where to clone the repository

    Returns:
        bool: True if cloning was successful, False otherwise
    """
    try:
        subprocess.run(
            ["git", "clone", f"https://github.com/{repo_name}.git", str(clone_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logging.info("Successfully cloned %s", repo_name)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Failed to clone %s: %s", repo_name, e.stderr.strip())
        return False
    except Exception as e:
        logging.error("Failed to clone %s: %s", repo_name, str(e))
        return False


def handle_repository(repo_name, clone_path):
    """
    Handle repository cloning or updating based on whether it already exists.

    Args:
        repo_name (str): Name of the repository in format "owner/repo"
        clone_path (Path): Path where to clone/update the repository

    Returns:
        bool: True if operation was successful, False otherwise
    """
    if clone_path.exists():
        if is_git_repo(clone_path):
            logging.info("Repository %s already exists, updating...", repo_name)
            return pull_latest_changes(clone_path, repo_name)
        else:
            logging.warning(
                "Directory %s exists but is not a valid Git repository. Removing and cloning fresh.",
                clone_path,
            )
            try:
                # Remove invalid repository directory
                import shutil

                shutil.rmtree(clone_path)
                # Clone repository
                return clone_repository(repo_name, clone_path)
            except Exception as e:
                logging.error(
                    "Failed to remove invalid repository directory %s: %s",
                    clone_path,
                    str(e),
                )
                return False
    else:
        # Repository doesn't exist, clone it
        return clone_repository(repo_name, clone_path)


def main():
    """Main function to process repositories and clone those with sufficient stars."""
    logging.basicConfig(
        format="%(asctime)s (PID %(process)d) [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Clone repositories from CSV file.")
    parser.add_argument(
        "--control",
        action="store_true",
        help="Clone control repositories instead of treatment repositories",
    )
    args = parser.parse_args()

    # Ensure clone directory exists
    ensure_dir(CLONE_DIR)
    ensure_dir(CONTROL_CLONE_DIR)

    if not args.control:
        # Read the CSV file
        try:
            df = pd.read_csv(REPOS_CSV)
            logging.info("%d repos from %s", len(df), REPOS_CSV)
        except Exception as e:
            logging.error("Failed to read CSV file: %s", str(e))
            return

        # Filter repositories with 10 or more stars
        repos_to_clone = df[df["repo_stars"] >= MIN_STARS]
        logging.info("Found %d repos with %d+ stars", len(repos_to_clone), MIN_STARS)
    else:
        try:
            df = pd.read_csv(CONTROL_REPOS_CSV)
            logging.info("%d controls from %s", len(df), CONTROL_REPOS_CSV)
        except Exception as e:
            logging.error("Failed to read CSV file: %s", str(e))
            return

        raise NotImplementedError("Control repositories not implemented yet")

    # Clone repositories
    success_count = 0
    update_count = 0
    start_time = datetime.now()

    logging.info("Starting repository processing at %s", start_time.isoformat())

    for idx, repo in repos_to_clone.iterrows():
        repo_name = repo["repo_name"]
        clone_path = CLONE_DIR / repo_name.replace("/", "_")

        # Check if repository exists before handling
        repo_existed = clone_path.exists() and is_git_repo(clone_path)

        if handle_repository(repo_name, clone_path):
            success_count += 1
            if repo_existed:
                update_count += 1

        # Log progress periodically
        if (idx + 1) % 10 == 0:
            logging.info(
                "Progress: %d/%d repositories processed", idx + 1, len(repos_to_clone)
            )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logging.info(
        "Processing complete. Successfully cloned/updated %d out of %d repositories",
        success_count,
        len(repos_to_clone),
    )
    logging.info("Operation took %.2f seconds", duration)

    # Print summary
    logging.info("\nRepository Processing Summary:")
    logging.info("Total repositories to process: %d", len(repos_to_clone))
    logging.info("Successfully cloned/updated: %d", success_count)
    logging.info("Newly cloned: %d", success_count - update_count)
    logging.info("Updated existing: %d", update_count)
    logging.info("Failed: %d", len(repos_to_clone) - success_count)


if __name__ == "__main__":
    main()
