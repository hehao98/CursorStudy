import logging
import os

import pandas as pd
from dotenv import load_dotenv
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

QUERY = """
query($first: Int, $after: String) {
  securityVulnerabilities(ecosystem: NPM, first: $first, after: $after)  {
    edges {
      node {
        advisory {
          ghsaId
          summary,
          description,
        },
        package {
          ecosystem,
          name
        }
        vulnerableVersionRange
      }
    }
    pageInfo {
      endCursor,
      hasNextPage
    },
    totalCount,
  },
}
"""


def get_github_token() -> str:
    load_dotenv(override=True)
    token = os.getenv("GITHUB_TOKEN")
    if token is None:
        logging.error("GitHub token not provided. Add GITHUB_TOKEN to your .env file.")
        raise ValueError(
            "GitHub token not provided. Add GITHUB_TOKEN to your .env file."
        )
    logging.info("GitHub token loaded from environment.")
    return token


def fetch_github_advisories() -> pd.DataFrame:
    logging.info("Starting fetch of GitHub advisories.")
    token = get_github_token()
    transport = AIOHTTPTransport(
        url="https://api.github.com/graphql",
        headers={"Authorization": "bearer " + token},
        ssl=False,
    )
    client = Client(transport=transport, fetch_schema_from_transport=True)

    query = gql(QUERY)

    after, has_next = None, True
    results = []
    page = 0
    while has_next:
        logging.info("Requesting advisories page %d (after=%s)", page + 1, after)
        result = client.execute(query, variable_values={"first": 100, "after": after})
        for node in result["securityVulnerabilities"]["edges"]:
            node = node["node"]
            results.append(
                {
                    "ghsaId": node["advisory"]["ghsaId"],
                    "package": node["package"]["name"],
                    "vulnerable_versions": node["vulnerableVersionRange"],
                    "summary": node["advisory"]["summary"],
                    "description": node["advisory"]["description"],
                }
            )
        has_next = result["securityVulnerabilities"]["pageInfo"]["hasNextPage"]
        after = result["securityVulnerabilities"]["pageInfo"]["endCursor"]
        page += 1
    logging.info("Fetched %d advisories.", len(results))
    return pd.DataFrame(results)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
    df = fetch_github_advisories()
    output_path = "data/github.npm.advisories.csv"
    df.to_csv(output_path, index=False)
    logging.info("Advisories saved to %s", output_path)


if __name__ == "__main__":
    main()
