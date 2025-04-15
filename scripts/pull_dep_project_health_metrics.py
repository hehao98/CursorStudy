#!/usr/bin/env python3
"""
Script to pull vulnerability data for packages using the Snyk API.
Includes caching functionality to avoid redundant API calls.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv

# Constants
CACHE_FILE = Path(__file__).parent.parent / "data" / "package_vulnerability_snyk_data.json"
SNYK_API_BASE = "https://snyk.io/api/v1"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_cache() -> Dict:
    """Load the vulnerability cache from disk."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logging.error("Cache file exists but is not valid JSON. Creating new cache.")
            return {}
    return {}

def save_cache(cache: Dict) -> None:
    """Save the vulnerability cache to disk."""
    # Ensure the data directory exists
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_vulnerabilities(package_name: str, version: str) -> List[Dict]:
    """
    Get vulnerabilities for a specific package version from Snyk API.
    
    Args:
        package_name: Name of the package
        version: Version of the package
    
    Returns:
        List of vulnerability dictionaries
    """
    # Load environment variables
    load_dotenv()
    api_token = os.getenv('SNYK_API_TOKEN')
    if not api_token:
        raise ValueError("SNYK_API_TOKEN not found in environment variables")
    
    logging.info("Using API token: %s...%s", api_token[:4], api_token[-4:])
    
    headers = {
        'Authorization': f'token {api_token}',
        'Content-Type': 'application/json',
    }

    # Construct the API endpoint URL
    endpoint = f"{SNYK_API_BASE}/test/npm/{package_name}/{version}"
    logging.info(f"Making request to: {endpoint}")
    logging.info(f"Request headers: {headers}")

    try:
        response = requests.post(endpoint, headers=headers, json={})
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            logging.error(f"Error response body: {response.text}")
        
        response.raise_for_status()
        data = response.json()
        
        # Extract vulnerabilities from response
        return data.get('issues', {}).get('vulnerabilities', [])
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching vulnerability data: {str(e)}")
        return []

def get_package_vulnerabilities(package_name: str, version: str) -> List[Dict]:
    """
    Get vulnerabilities for a package version, using cache if available.
    
    Args:
        package_name: Name of the package
        version: Version of the package
    
    Returns:
        List of vulnerability dictionaries
    """
    # Load cache
    cache = load_cache()
    
    # Check if we have cached data
    if package_name in cache and version in cache[package_name]:
        logging.info(f"Found cached vulnerability data for {package_name}@{version}")
        return cache[package_name][version]
    
    # If not in cache, fetch from API
    logging.info(f"Fetching vulnerability data for {package_name}@{version} from Snyk API")
    vulnerabilities = get_vulnerabilities(package_name, version)
    
    # Update cache
    if package_name not in cache:
        cache[package_name] = {}
    cache[package_name][version] = vulnerabilities
    
    # Save updated cache
    save_cache(cache)
    
    return vulnerabilities

def main():
    """Main function to run the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Get package vulnerabilities from Snyk API')
    parser.add_argument('package_name', help='Name of the package')
    parser.add_argument('version', help='Version of the package')
    
    args = parser.parse_args()
    
    try:
        vulnerabilities = get_package_vulnerabilities(args.package_name, args.version)
        if vulnerabilities:
            print(f"\nFound {len(vulnerabilities)} vulnerabilities for {args.package_name}@{args.version}:")
            for vuln in vulnerabilities:
                print(f"\n- {vuln.get('title', 'Untitled Vulnerability')}")
                print(f"  Severity: {vuln.get('severity', 'Unknown')}")
                print(f"  CVE: {vuln.get('identifiers', {}).get('CVE', ['N/A'])[0]}")
        else:
            print(f"\nNo vulnerabilities found for {args.package_name}@{args.version}")
    
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
