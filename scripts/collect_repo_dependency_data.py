"""
Script to collect package.json dependency data on the latest commits of each week.

This script:
1. Reads the weekly repository stats from ts_repos.csv
2. For each repository and week, pulls the package.json file on the latest commit
3. Collects and stores the dependency data (both aggregate weekly values and all raw dependency data for technical lag calculations)
"""

import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import git
import pandas as pd
import requests
import semver
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Metrics to collect from SonarQube
METRICS_OF_INTEREST = [
    "num_dependencies_total",  # technical debt
    "num_normal_dependencies",
    "num_dev_dependencies",
    "num_peer_dependencies",
]

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
CLONE_DIR = SCRIPT_DIR.parent.parent / "CursorRepos"
TS_REPOS_CSV = DATA_DIR / "ts_repos_weekly.csv"
REPOS_CSV = DATA_DIR / "repos.csv"  # Add path for repos data
WEEKS_INTERVAL = 52  # It may be too costly to analyze all weeks
NUM_PROCESSES = 16  # Number of processes to use for parallel processing
DEPENDENCY_DECLARATIONS_CSV = DATA_DIR / "dependency_declarations.csv"
UNIQUE_PACKAGE_VERSIONS_CSV = DATA_DIR / "unique_dependency_package_versions.csv"


# Initialize CSV files and data structures
def setup_csv_files() -> Tuple[Dict, set]:
    """Initialize and set up the CSV files for dependency data."""
    # Set up unique package versions CSV
    if not os.path.exists(UNIQUE_PACKAGE_VERSIONS_CSV):
        with open(UNIQUE_PACKAGE_VERSIONS_CSV, "w") as f:
            f.write("package,version\n")
        unique_package_versions = {}
    else:
        # Load existing unique package versions into memory
        unique_package_versions_df = pd.read_csv(UNIQUE_PACKAGE_VERSIONS_CSV)
        unique_package_versions = {
            (row["package"], row["version"]): True
            for _, row in unique_package_versions_df.iterrows()
        }

    # Set up dependency declarations CSV
    if not os.path.exists(DEPENDENCY_DECLARATIONS_CSV):
        with open(DEPENDENCY_DECLARATIONS_CSV, "w") as f:
            f.write(
                "repo_name,week,commit_sha,dependency,version,concreteVersion,dependency_type\n"
            )
        project_names = set()
    else:
        # Check if we need to update the CSV format to include concreteVersion
        with open(DEPENDENCY_DECLARATIONS_CSV, "r") as f:
            header = f.readline().strip()

        # Backup the old file before making any changes
        import shutil

        backup_file = f"{DEPENDENCY_DECLARATIONS_CSV}.bak"
        if not os.path.exists(backup_file):
            shutil.copy(DEPENDENCY_DECLARATIONS_CSV, backup_file)

        # Read the old data
        old_data = pd.read_csv(DEPENDENCY_DECLARATIONS_CSV)

        # If concreteVersion not in header, add it
        if "concreteVersion" not in header:
            # Determine the correct column name for dependency type
            dep_type_col = None
            for possible_col in ["dependency_type", "dependencyType", "type"]:
                if possible_col in old_data.columns:
                    dep_type_col = possible_col
                    break

            if dep_type_col is None:
                # If no dependency type column exists, create a new file with updated structure
                with open(DEPENDENCY_DECLARATIONS_CSV, "w") as new_f:
                    new_f.write(
                        "repo_name,week,commit_sha,dependency,version,concreteVersion,dependency_type\n"
                    )
            else:
                # Create a new file with updated header
                with open(DEPENDENCY_DECLARATIONS_CSV, "w") as new_f:
                    new_f.write(
                        "repo_name,week,commit_sha,dependency,version,concreteVersion,dependency_type\n"
                    )

                    # Write the old data with concreteVersion same as version
                    for _, row in old_data.iterrows():
                        new_f.write(
                            f"{row['repo_name']},{row['week']},{row['commit_sha']},{row['dependency']},{row['version']},{row['version']},{row[dep_type_col]}\n"
                        )

        # Load existing project names from dependency declarations
        dependency_declarations_df = pd.read_csv(DEPENDENCY_DECLARATIONS_CSV)
        project_names = set(dependency_declarations_df["repo_name"].unique())

    return unique_package_versions, project_names


# Initialize data structures
UNIQUE_PACKAGE_VERSIONS, PROJECT_NAMES = setup_csv_files()


def week_to_date(week: str) -> datetime:
    """
    Convert a week string (YYYY-WXX) to the last day of that week.

    Args:
        week: Week identifier (YYYY-WXX format)

    Returns:
        datetime object representing the last day of the week
    """
    year, week_num = week.split("-W")
    # Create a date for the first day of the year
    first_day = datetime(int(year), 1, 1)
    # Adjust to get the first day of the first week
    if first_day.weekday() <= 3:
        first_week_day = first_day - timedelta(days=first_day.weekday())
    else:
        first_week_day = first_day + timedelta(days=7 - first_day.weekday())
    # Calculate the start of our target week
    target_week_start = first_week_day + timedelta(weeks=int(week_num) - 1)
    # Return the end of the week
    return target_week_start + timedelta(days=6)


def resolve_semantic_version(
    package_name: str, semantic_version: str, week: str
) -> str:
    """
    Resolve a semantic version to a concrete version available at a specific week.

    Args:
        package_name: Name of the npm package
        semantic_version: The semantic version specification (e.g., ^9.19.0)
        week: Week identifier (YYYY-WXX format)

    Returns:
        The concrete version that would have been used (or the original version if resolution fails)
    """
    try:
        # Convert week to a date (end of week)
        week_end_date = week_to_date(week)

        # Query the npm registry for package info
        registry_url = f"https://registry.npmjs.org/{package_name}"
        response = requests.get(registry_url)

        if response.status_code != 200:
            logging.warning(
                "Failed to fetch npm info for %s: HTTP %s",
                package_name,
                response.status_code,
            )
            return semantic_version

        package_data = response.json()

        # Handle case where package not found
        if "versions" not in package_data or "time" not in package_data:
            return semantic_version

        valid_versions = []

        # Extract version numbers and their publish dates
        for version in package_data["versions"].keys():
            if version in package_data["time"]:
                try:
                    publish_date_str = package_data["time"][version]

                    # Handle different date formats
                    if publish_date_str.endswith("Z"):
                        # ISO format with Z
                        publish_date_str = publish_date_str[:-1]
                        publish_date = datetime.fromisoformat(publish_date_str)
                    elif "T" in publish_date_str and "+" in publish_date_str:
                        # ISO format with timezone offset
                        publish_date = datetime.fromisoformat(publish_date_str)
                        # Convert to naive datetime for comparison
                        publish_date = publish_date.replace(tzinfo=None)
                    else:
                        # Try standard format
                        publish_date = datetime.fromisoformat(publish_date_str)

                    # Only include versions published before or during the target week
                    if publish_date <= week_end_date:
                        try:
                            # Check if this is a valid semver
                            semver.VersionInfo.parse(version)
                            valid_versions.append(version)
                        except (ValueError, AttributeError):
                            # Skip non-semver versions
                            continue
                except (ValueError, TypeError, AttributeError) as e:
                    logging.debug(
                        "Error parsing date for %s@%s: %s",
                        package_name,
                        version,
                        str(e),
                    )
                    continue

        if not valid_versions:
            return semantic_version

        # Parse the requested version range
        if semantic_version.startswith("^"):
            # Caret range: compatible with specified version, allowing all updates except major (if >= 1.0.0)
            # or minor (if < 1.0.0)
            base_version_str = semantic_version[1:]
            try:
                base_version = semver.VersionInfo.parse(base_version_str)
                compatible_versions = []

                for v in valid_versions:
                    try:
                        v_parsed = semver.VersionInfo.parse(v)

                        # Apply ^version rules
                        if base_version.major >= 1:
                            # For versions >= 1.0.0, allow patches and minor updates
                            if (
                                v_parsed.major == base_version.major
                                and v_parsed >= base_version
                            ):
                                compatible_versions.append(v)
                        else:
                            # For versions < 1.0.0, only allow patch updates
                            if (
                                v_parsed.major == 0
                                and v_parsed.minor == base_version.minor
                                and v_parsed >= base_version
                            ):
                                compatible_versions.append(v)
                    except (ValueError, AttributeError):
                        continue

                if compatible_versions:
                    # Sort by version and return the highest compatible version
                    return max(
                        compatible_versions, key=lambda x: semver.VersionInfo.parse(x)
                    )
                return semantic_version
            except (ValueError, AttributeError):
                return semantic_version

        elif semantic_version.startswith("~"):
            # Tilde range: allow patch updates but not minor or major
            base_version_str = semantic_version[1:]
            try:
                base_version = semver.VersionInfo.parse(base_version_str)
                compatible_versions = []

                for v in valid_versions:
                    try:
                        v_parsed = semver.VersionInfo.parse(v)

                        # Apply ~version rules (same major and minor, patch can be higher)
                        if (
                            v_parsed.major == base_version.major
                            and v_parsed.minor == base_version.minor
                            and v_parsed >= base_version
                        ):
                            compatible_versions.append(v)
                    except (ValueError, AttributeError):
                        continue

                if compatible_versions:
                    # Sort by version and return the highest compatible version
                    return max(
                        compatible_versions, key=lambda x: semver.VersionInfo.parse(x)
                    )
                return semantic_version
            except (ValueError, AttributeError):
                return semantic_version

        elif (
            ">" in semantic_version
            or "<" in semantic_version
            or "-" in semantic_version
        ):
            # For complex ranges, simplify by taking the highest version available at the time
            # This is a simplification - proper range handling would need more sophisticated logic
            if valid_versions:
                return max(valid_versions, key=lambda x: semver.VersionInfo.parse(x))
            return semantic_version

        else:
            # Exact version match
            if semantic_version in valid_versions:
                return semantic_version

            # Try to find closest match for exact version
            try:
                requested = semver.VersionInfo.parse(semantic_version)

                # Find versions that satisfy the requirement
                valid_parsed = []
                for v in valid_versions:
                    try:
                        parsed = semver.VersionInfo.parse(v)
                        # Add tuples of (version string, parsed version object)
                        valid_parsed.append((v, parsed))
                    except (ValueError, AttributeError):
                        continue

                # Sort versions in descending order
                valid_parsed.sort(key=lambda x: x[1], reverse=True)

                # Find highest version that's <= requested version
                for v, parsed in valid_parsed:
                    if parsed <= requested:
                        return v
                return semantic_version
            except (ValueError, AttributeError):
                return semantic_version

        # If we get here, no suitable version was found
        return semantic_version

    except Exception as e:
        logging.error(
            "Failed to resolve version for %s@%s: %s",
            package_name,
            semantic_version,
            str(e),
        )
        return semantic_version


def parse_and_store_dependency_declarations(
    repo_name: str, week: str, commit_hash: str, results: Dict
) -> None:
    """
    Parse and store dependency declarations for a specific repository and week.

    Args:
        repo_name: Name of the repository
        week: Week identifier (YYYY-WXX format)
        commit_hash: Git commit hash
        results: Dictionary containing dependency information
            - 'dependencies': normal dependencies
            - 'devDependencies': development dependencies
            - 'peerDependencies': peer dependencies
    """
    global UNIQUE_PACKAGE_VERSIONS

    # Map of dictionary keys to dependency types
    dep_type_map = {
        "dependencies": "normal",
        "devDependencies": "dev",
        "peerDependencies": "peer",
    }

    # Rate limiting for npm API (to avoid hitting rate limits)
    api_calls = 0

    # Open dependency declarations CSV in append mode
    with open(DEPENDENCY_DECLARATIONS_CSV, "a") as declarations_file:
        # Process each type of dependency
        for dep_dict_key, dep_type in dep_type_map.items():
            dependencies = results.get(dep_dict_key, {})

            # Process each dependency and its version
            for dependency, version in dependencies.items():
                # Resolve semantic version to concrete version
                api_calls += 1
                if api_calls % 10 == 0:
                    time.sleep(1)  # Sleep to avoid rate limiting

                concrete_version = resolve_semantic_version(dependency, version, week)
                # print("the version they declared is", version, "and the concrete version is", concrete_version)
                # Write to dependency declarations CSV
                declarations_file.write(
                    f"{repo_name},{week},{commit_hash},{dependency},{version},{concrete_version},{dep_type}\n"
                )

                # Check if this package-concrete_version pair is already in our unique versions
                if (dependency, concrete_version) not in UNIQUE_PACKAGE_VERSIONS:
                    # Add to memory
                    UNIQUE_PACKAGE_VERSIONS[(dependency, concrete_version)] = True

                    # Write to unique package versions CSV
                    with open(UNIQUE_PACKAGE_VERSIONS_CSV, "a") as versions_file:
                        versions_file.write(f"{dependency},{concrete_version}\n")


def pull_package_json_file_from_repo(
    repo_path: Path, commit_hash: str
) -> Optional[Dict]:
    """
    Pull package.json data from a specific commit.

    Args:
        repo_path: Path to the repository
        commit_hash: Git commit hash to analyze

    Returns:
        dict: Dictionary with dependency information, or None if unable to parse
    """
    try:
        # Initialize dependency_dict with empty objects
        dependency_dict = {
            "dependencies": {},
            "devDependencies": {},
            "peerDependencies": {},
        }

        # Initialize git repo without checkout
        repo = git.Repo(str(repo_path))

        # Try to get package.json content directly from git without checking out
        try:
            # Get file content at specific commit without checking out
            file_content = repo.git.show(f"{commit_hash}:package.json")
            logging.info(
                "Successfully retrieved package.json content from commit %s",
                commit_hash,
            )

            # Parse JSON content
            package_data = json.loads(file_content)

            # Extract dependency sections if they exist
            dependencies = package_data.get("dependencies", {})
            dev_dependencies = package_data.get("devDependencies", {})
            peer_dependencies = package_data.get("peerDependencies", {})

            # creating dictionary with all dependency data
            dependency_dict = {
                "dependencies": dependencies,
                "devDependencies": dev_dependencies,
                "peerDependencies": peer_dependencies,
            }

            return dependency_dict

        except git.exc.GitCommandError:
            # If the file doesn't exist in the commit
            print(
                "package.json does not exist in this repository at commit", commit_hash
            )
            return dependency_dict

    except Exception as e:
        logging.error(
            "Error during collection of package.json for %s at %s: %s",
            repo_path,
            commit_hash,
            str(e),
        )
        return None


def process_repository(
    ts_df: pd.DataFrame, repos_df: pd.DataFrame, repo_name: str
) -> pd.DataFrame:
    """
    Process a single repository's analysis.

    Args:
        ts_df: Time series dataframe
        repos_df: Repository information dataframe
        repo_name: Name of the repository to process

    Returns:
        pd.DataFrame: Updated time series dataframe for this repository
    """
    repo_path = CLONE_DIR / repo_name.replace("/", "_")
    if not repo_path.exists():
        logging.warning("Repository %s not found at %s", repo_name, repo_path)
        return ts_df[ts_df["repo_name"] == repo_name]

    # Get adoption week from repos data
    repo_info = repos_df[repos_df["repo_name"] == repo_name]
    if repo_info.empty or pd.isna(repo_info["repo_cursor_adoption"].iloc[0]):
        logging.warning("No adoption found for %s, skipping", repo_name)
        return ts_df[ts_df["repo_name"] == repo_name]

    adoption = repo_info["repo_cursor_adoption"].iloc[0]
    start_week = (adoption - pd.Timedelta(weeks=WEEKS_INTERVAL)).strftime("%Y-W%W")
    end_week = (adoption + pd.Timedelta(weeks=WEEKS_INTERVAL)).strftime("%Y-W%W")
    logging.info("Processing %s from %s to %s", repo_name, start_week, end_week)

    # Get repository data
    repo_df = ts_df[ts_df["repo_name"] == repo_name].copy()

    # Filter weeks before and after adoption
    weeks_in_range = sorted(
        repo_df[(repo_df["week"] >= start_week) & (repo_df["week"] <= end_week)][
            "week"
        ].unique()
    )
    print("weeks in range", weeks_in_range)

    # Process each week's latest commit in chronological order
    for week in weeks_in_range:
        # Get the index in the repository dataframe
        row_idx = repo_df[repo_df["week"] == week].index[0]
        commit_hash = repo_df.loc[row_idx, "latest_commit"]

        print(row_idx, commit_hash)

        if not commit_hash:
            logging.warning("No commit hash for %s at %s", repo_name, week)
            continue

        # Add check here to see if we've already collected dependency data for this repo TODO
        print("attempting to pull dependencies from package.json for this commit")
        # pulling dependencies from package.json for this commit
        results = pull_package_json_file_from_repo(repo_path, commit_hash)

        # add check for if the results are None meaning we could not collect the package.json data
        if results is None:
            logging.error(
                "Failed to extract package.json data for %s at %s",
                repo_name,
                commit_hash,
            )
            return ts_df[ts_df["repo_name"] == repo_name]

        # save all dependency declarations in other function and create unique list of packages and versions for next data collection
        parse_and_store_dependency_declarations(repo_name, week, commit_hash, results)

        # calculating the metrics of interest
        metrics = {
            "num_dependencies_total": len(results["dependencies"])
            + len(results["devDependencies"])
            + len(results["peerDependencies"]),
            "num_normal_dependencies": len(results["dependencies"]),
            "num_dev_dependencies": len(results["devDependencies"]),
            "num_peer_dependencies": len(results["peerDependencies"]),
        }

        if metrics:
            for metric, value in metrics.items():
                repo_df.loc[row_idx, metric] = value
        logging.info("Metrics for %s at %s: %s", repo_name, week, metrics)

    return repo_df


def main() -> None:
    """Main function to run project.json dependency data collection on repositories."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set up CSV files and data structures
    global UNIQUE_PACKAGE_VERSIONS, PROJECT_NAMES
    UNIQUE_PACKAGE_VERSIONS, PROJECT_NAMES = setup_csv_files()

    # Read repository time series data and adoption data
    ts_df = pd.read_csv(TS_REPOS_CSV)
    repos_df = pd.read_csv(REPOS_CSV)

    # Convert cursor_adoption_week to datetime
    repos_df["repo_cursor_adoption"] = pd.to_datetime(repos_df["repo_cursor_adoption"])

    # Create columns for metrics if they don't exist
    for col in METRICS_OF_INTEREST:
        if col not in ts_df.columns:
            ts_df[col] = None

    # Get unique repository names
    repo_names = ts_df["repo_name"].unique()

    # pull name of first repo as example TODO
    # repo_names[2] doesn't have package.json
    # repo_name = repo_names[1]
    # pulling results ror repo
    # results = process_repository(ts_df, repos_df, repo_name)
    with mp.Pool(NUM_PROCESSES) as pool:
        args = [(ts_df, repos_df, repo_name) for repo_name in repo_names]
        results = pool.starmap(process_repository, args, chunksize=1)

    updated_df = pd.concat(results)
    updated_df.sort_values(by=["repo_name", "week"]).to_csv(TS_REPOS_CSV, index=False)
    logging.info("Updated metrics saved to %s", TS_REPOS_CSV)


if __name__ == "__main__":
    main()
