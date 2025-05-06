#!/usr/bin/env python3
"""
Script to collect number of dependencies per month for repositories with a package.json file.

For each row in data/ts_repos_monthly.csv (or data/ts_repos_control_monthly.csv with --control):
- For each (repo_name, month, latest_commit), check if package.json exists at that commit
- If so, count dependencies and check for vulnerabilities
- Add dependency metrics columns to the same CSV in-place
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import git
import nodesemver
import pandas as pd
import requests
import semver

nodesemver.logger.setLevel(logging.WARNING)

# Constants
TS_REPOS_CSV = Path(__file__).parent.parent / "data" / "ts_repos_monthly.csv"
TS_REPOS_CONTROL_CSV = (
    Path(__file__).parent.parent / "data" / "ts_repos_control_monthly.csv"
)
GITHUB_ADVISORIES_CSV = (
    Path(__file__).parent.parent / "data" / "github_npm_advisories.csv"
)
CURSOR_REPOS_DIR = Path(__file__).parent.parent.parent / "CursorRepos"
CONTROL_REPOS_DIR = Path(__file__).parent.parent.parent / "ControlRepos"

DEPENDENCY_COLS = [
    "num_dependencies_total",
    "num_normal_dependencies",
    "num_dev_dependencies",
    "num_peer_dependencies",
    "num_vulnerable_dependencies",
    "average_technical_lag",
]


class NpmRegistryClient:
    """Client for querying npm registry with caching"""

    def __init__(self):
        self.cache = {}  # Cache package data in memory
        self.request_count = 0
        self.cache_hits = 0

    def get_package_data(self, package_name: str) -> Optional[Dict]:
        """Get package data from npm registry with caching"""
        # Check cache first
        if package_name in self.cache:
            self.cache_hits += 1
            return self.cache[package_name]

        # Rate limiting to avoid npm registry throttling
        self.request_count += 1
        if self.request_count % 20 == 0:
            time.sleep(1)

        # Query npm registry
        try:
            registry_url = f"https://registry.npmjs.org/{package_name}"
            response = requests.get(registry_url, timeout=5)

            if response.status_code == 200:
                package_data = response.json()
                # Cache the result
                self.cache[package_name] = package_data
                return package_data
            else:
                logging.error(
                    "Failed to fetch npm data for %s: HTTP %s",
                    package_name,
                    response.status_code,
                )
                return None
        except Exception as e:
            logging.error("Error fetching npm data for %s: %s", package_name, str(e))
            return None

    def get_version_release_date(
        self, package_name: str, version: str
    ) -> Optional[datetime]:
        """Get the release date of a specific package version"""
        package_data = self.get_package_data(package_name)
        if (
            not package_data
            or "time" not in package_data
            or version not in package_data["time"]
        ):
            return None

        try:
            date_str = package_data["time"][version]
            # Handle different date formats
            if date_str.endswith("Z"):
                date_str = date_str[:-1]
            return datetime.fromisoformat(date_str)
        except Exception as e:
            logging.error(
                "Error parsing date for %s@%s: %s", package_name, version, str(e)
            )
            return None

    def get_latest_version(self, package_name: str) -> Optional[str]:
        """Get the latest version of a package"""
        package_data = self.get_package_data(package_name)
        if (
            not package_data
            or "dist-tags" not in package_data
            or "latest" not in package_data["dist-tags"]
        ):
            return None
        return package_data["dist-tags"]["latest"]

    def get_versions_before_date(
        self, package_name: str, reference_date: datetime
    ) -> list[tuple[str, datetime]]:
        """Get all versions of a package released before the given date, with their release dates"""
        package_data = self.get_package_data(package_name)
        if not package_data or "time" not in package_data:
            return []

        # Filter versions by release date
        valid_versions = []
        for version, date_str in package_data["time"].items():
            # Skip created, modified and other metadata entries
            if version in ["created", "modified"]:
                continue

            try:
                if date_str.endswith("Z"):
                    date_str = date_str[:-1]
                release_date = datetime.fromisoformat(date_str)
                # Only include versions released before the reference date
                if release_date <= reference_date and version in package_data.get(
                    "versions", {}
                ):
                    valid_versions.append((version, release_date))
            except Exception:
                # Skip invalid dates
                continue

        return valid_versions

    def get_latest_version_before_date(
        self, package_name: str, reference_date: datetime
    ) -> Optional[str]:
        """Get the latest version of a package available before the given date"""
        valid_versions = self.get_versions_before_date(package_name, reference_date)

        if not valid_versions:
            return None

        # Sort by date (newest first) and return the latest version
        valid_versions.sort(key=lambda x: x[1], reverse=True)
        return valid_versions[0][0]

    def get_technical_lag_days(
        self, package_name: str, current_version: str, commit_date: datetime
    ) -> Optional[float]:
        """Calculate technical lag in days between current version and latest version available at commit date"""
        try:
            package_data = self.get_package_data(package_name)
            if (
                not package_data
                or "versions" not in package_data
                or "time" not in package_data
            ):
                logging.info(
                    "Skipping package %s: no package data",
                    package_name,
                )
                return None

            # Get the latest version available at commit date
            latest_version = self.get_latest_version_before_date(
                package_name, commit_date
            )
            if not latest_version:
                logging.info(
                    "Skipping package %s: no latest version before %s",
                    package_name,
                    commit_date.isoformat(),
                )
                return None

            # Get all available versions before the commit date
            valid_versions = self.get_versions_before_date(package_name, commit_date)
            if not valid_versions:
                logging.info(
                    "Skipping package %s: no available versions before %s",
                    package_name,
                    commit_date.isoformat(),
                )
                return None

            # Extract just the version strings for semver comparison
            available_versions = [v[0] for v in valid_versions]

            # Find highest version satisfying the range constraint
            effective_version = nodesemver.max_satisfying(
                available_versions, current_version
            )

            if not effective_version:
                logging.info(
                    "Skipping package %s: no version satisfying %s before %s",
                    package_name,
                    current_version,
                    commit_date.isoformat(),
                )
                return None

            if effective_version == latest_version:
                return 0

            # Get release dates
            effective_date = self.get_version_release_date(
                package_name, effective_version
            )
            latest_date = self.get_version_release_date(package_name, latest_version)

            if not effective_date or not latest_date:
                logging.info(
                    "Skipping package %s: no effective or latest date",
                    package_name,
                )
                return None

            # Calculate lag in days
            lag_days = (latest_date - effective_date).total_seconds() / (60 * 60 * 24)
            return max(0, lag_days)  # Ensure lag is non-negative

        except Exception as e:
            logging.error(
                "Error calculating technical lag for %s@%s: %s",
                package_name,
                current_version,
                str(e),
            )
            return None


class VulnerabilityChecker:
    """Checks if a package and version are vulnerable according to GitHub advisories."""

    def __init__(self):
        self.advisories = {}
        try:
            df = pd.read_csv(GITHUB_ADVISORIES_CSV)
            for _, row in df.iterrows():
                package = row["package"]
                vulnerable_versions = row["vulnerable_versions"]
                if package not in self.advisories:
                    self.advisories[package] = []
                self.advisories[package].append(vulnerable_versions)
            logging.info(
                "Loaded %d vulnerable packages from advisories", len(self.advisories)
            )
        except Exception as e:
            logging.error("Failed to load GitHub advisories: %s", str(e))
            self.advisories = {}

    def is_vulnerable(self, package: str, version: str) -> bool:
        """Check if a specific package and version is vulnerable."""
        if package not in self.advisories:
            return False

        # Try to parse the version as semver
        try:
            parsed_version = semver.VersionInfo.parse(version.lstrip("^~=v"))
        except ValueError:
            # If we can't parse the version, be conservative and return False
            return False

        for vulnerable_versions in self.advisories[package]:
            if self._matches_vulnerability_pattern(parsed_version, vulnerable_versions):
                return True
        return False

    def _matches_vulnerability_pattern(
        self, parsed_version: semver.VersionInfo, pattern: str
    ) -> bool:
        """Determine if a version matches a vulnerability pattern using semver."""
        # Exact version match (remove v prefix if present)
        if pattern.startswith("v"):
            pattern = pattern[1:]
        if pattern == str(parsed_version):
            return True

        try:
            # Handle common comparison operators
            if pattern.startswith("<="):
                pattern_version = pattern[2:].strip().lstrip("v")
                return parsed_version <= semver.VersionInfo.parse(pattern_version)

            elif pattern.startswith("<"):
                pattern_version = pattern[1:].strip().lstrip("v")
                return parsed_version < semver.VersionInfo.parse(pattern_version)

            elif pattern.startswith(">="):
                pattern_version = pattern[2:].strip().lstrip("v")
                return parsed_version >= semver.VersionInfo.parse(pattern_version)

            elif pattern.startswith(">"):
                pattern_version = pattern[1:].strip().lstrip("v")
                return parsed_version > semver.VersionInfo.parse(pattern_version)

            # Version ranges like ">=1.0.0 <2.0.0"
            elif " " in pattern:
                # Split by whitespace to get different range components
                range_parts = pattern.split()
                matches_all = True

                for range_part in range_parts:
                    # Skip logical operators like "||" or "&&"
                    if range_part in ["||", "&&"]:
                        continue

                    # Parse each range component
                    if range_part.startswith("<="):
                        version_str = range_part[2:].strip().lstrip("v")
                        if not (
                            parsed_version <= semver.VersionInfo.parse(version_str)
                        ):
                            matches_all = False
                            break
                    elif range_part.startswith("<"):
                        version_str = range_part[1:].strip().lstrip("v")
                        if not (parsed_version < semver.VersionInfo.parse(version_str)):
                            matches_all = False
                            break
                    elif range_part.startswith(">="):
                        version_str = range_part[2:].strip().lstrip("v")
                        if not (
                            parsed_version >= semver.VersionInfo.parse(version_str)
                        ):
                            matches_all = False
                            break
                    elif range_part.startswith(">"):
                        version_str = range_part[1:].strip().lstrip("v")
                        if not (parsed_version > semver.VersionInfo.parse(version_str)):
                            matches_all = False
                            break
                    elif range_part.startswith("="):
                        version_str = range_part[1:].strip().lstrip("v")
                        if not (
                            parsed_version == semver.VersionInfo.parse(version_str)
                        ):
                            matches_all = False
                            break

                return matches_all

            # Range with hyphen like "1.0.0 - 2.0.0"
            elif " - " in pattern:
                lower, upper = pattern.split(" - ")
                lower = lower.strip().lstrip("v")
                upper = upper.strip().lstrip("v")
                return parsed_version >= semver.VersionInfo.parse(
                    lower
                ) and parsed_version <= semver.VersionInfo.parse(upper)

            return False
        except (ValueError, AttributeError):
            # If we can't parse pattern versions, be conservative
            return False


def get_package_json_deps(
    repo_path: Path,
    commit_hash: str,
    vulnerability_checker: VulnerabilityChecker,
    npm_client: NpmRegistryClient,
) -> Optional[Dict[str, int]]:
    """
    At a given commit, return dependency counts from package.json, or None if not found/invalid.
    Also checks for vulnerabilities in dependencies and calculates technical lag.
    """
    try:
        repo = git.Repo(str(repo_path))
        try:
            # Get the commit date
            commit = repo.commit(commit_hash)
            commit_date = datetime.fromtimestamp(commit.committed_date)

            file_content = repo.git.show("%s:package.json" % commit_hash)
            package_data = json.loads(file_content)
            deps = package_data.get("dependencies", {})
            dev_deps = package_data.get("devDependencies", {})
            peer_deps = package_data.get("peerDependencies", {})

            # Count vulnerable dependencies
            vulnerable_count = 0

            # Calculate technical lag
            total_lag_days = 0
            valid_lag_measurements = 0

            # Process all dependencies
            for pkg_dict in [deps, dev_deps, peer_deps]:
                for pkg, version in pkg_dict.items():
                    # Check for vulnerabilities
                    if vulnerability_checker.is_vulnerable(pkg, version):
                        vulnerable_count += 1

                    # Calculate technical lag considering the commit date
                    lag_days = npm_client.get_technical_lag_days(
                        pkg, version, commit_date
                    )
                    if lag_days is not None:
                        total_lag_days += lag_days
                        valid_lag_measurements += 1

            # Compute average technical lag
            avg_technical_lag = (
                total_lag_days / valid_lag_measurements
                if valid_lag_measurements > 0
                else None
            )

            return {
                "num_dependencies_total": len(deps) + len(dev_deps) + len(peer_deps),
                "num_normal_dependencies": len(deps),
                "num_dev_dependencies": len(dev_deps),
                "num_peer_dependencies": len(peer_deps),
                "num_vulnerable_dependencies": vulnerable_count,
                "average_technical_lag": avg_technical_lag,
            }
        except git.exc.GitCommandError:
            return None
        except json.JSONDecodeError:
            return None
    except Exception as e:
        logging.error(
            "Error reading package.json for %s at %s: %s",
            str(repo_path),
            commit_hash,
            str(e),
        )
        return None


def process_repo_rows(
    repo_name: str,
    repo_rows: pd.DataFrame,
    vulnerability_checker: VulnerabilityChecker,
    npm_client: NpmRegistryClient,
    is_control: bool = False,
) -> pd.DataFrame:
    repo_path = (
        CONTROL_REPOS_DIR / repo_name.replace("/", "_")
        if is_control
        else CURSOR_REPOS_DIR / repo_name.replace("/", "_")
    )
    for idx, row in repo_rows.iterrows():
        commit_hash = row["latest_commit"]
        if (
            not repo_path.exists()
            or not isinstance(commit_hash, str)
            or not commit_hash
        ):
            for col in DEPENDENCY_COLS:
                repo_rows.at[idx, col] = None
            continue
        dep_counts = get_package_json_deps(
            repo_path, commit_hash, vulnerability_checker, npm_client
        )

        if dep_counts is not None:
            for col in DEPENDENCY_COLS:
                repo_rows.at[idx, col] = dep_counts[col]
        else:
            for col in DEPENDENCY_COLS:
                repo_rows.at[idx, col] = None

        logging.info("Dep counts %s %s: %s", repo_name, row["month"], dep_counts)
    return repo_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect monthly dependency counts for repos with package.json."
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Read and update data/ts_repos_control_monthly.csv instead of data/ts_repos_monthly.csv",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    input_csv = TS_REPOS_CONTROL_CSV if args.control else TS_REPOS_CSV

    try:
        ts_df = pd.read_csv(input_csv)
    except Exception as e:
        logging.error("Failed to read %s: %s", str(input_csv), str(e))
        return

    logging.info(
        "Processing %d repo-month-commit rows from %s",
        len(ts_df),
        str(input_csv),
    )

    # Initialize npm registry client and vulnerability checker
    npm_client = NpmRegistryClient()
    vulnerability_checker = VulnerabilityChecker()

    # Ensure dependency columns exist
    for col in DEPENDENCY_COLS:
        if col not in ts_df.columns:
            ts_df[col] = None

    # Process one repo at a time (sequentially)
    for repo_name, repo_rows in ts_df.groupby("repo_name"):
        updated_rows = process_repo_rows(
            repo_name,
            repo_rows.copy(),
            vulnerability_checker,
            npm_client,
            is_control=args.control,
        )
        ts_df.loc[updated_rows.index, DEPENDENCY_COLS] = updated_rows[DEPENDENCY_COLS]

    ts_df.to_csv(input_csv, index=False)
    logging.info("Updated dependency metrics in %s", str(input_csv))
    logging.info(
        "Npm API stats: %d requests, %d cache hits",
        npm_client.request_count,
        npm_client.cache_hits,
    )


if __name__ == "__main__":
    main()
