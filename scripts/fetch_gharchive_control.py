#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

CONTROL_REPO_QUERY = """
-- 0. Declare your target month (YYYYMM)
-- DECLARE target_month   STRING DEFAULT '202407';

WITH
  -- 1. Candidate repos: active in target_month & ≥10 stars ever
  candid_repos AS (
    SELECT
      repo.name AS repo_name
    FROM
      `githubarchive.month.*`
    WHERE
      type = 'WatchEvent'
      AND repo.name IN (
        SELECT DISTINCT repo.name
        FROM `githubarchive.month.*`
        WHERE _TABLE_SUFFIX = @target_month
      )
    GROUP BY
      repo_name
    HAVING
      COUNT(*) >= 10
  ),

  -- 2. All events for those repos up through the target month
  events AS (
    SELECT
      repo.name       AS repo_name,
      created_at,
      type,
      actor.id        AS actor_id,
      _TABLE_SUFFIX   AS ym
    FROM
      `githubarchive.month.*`
    WHERE
      _TABLE_SUFFIX <= @target_month
      AND repo.name IN (SELECT repo_name FROM candid_repos)
  ),

  -- 3. Extract the first‐of‐target date
  params AS (
    SELECT
      PARSE_DATE('%Y%m', @target_month) AS first_of_target
    FROM (SELECT @target_month)
  ),

  -- 4. Build 7 "period" rows:
  --    •  Six individual months (1..6 ago) → period_type='within'
  --    •  One "sum" bucket (7 months ago)     → period_type='sum'
  periods AS (
    -- the six most recent months
    SELECT
      FORMAT_DATE('%Y%m',
        DATE_SUB(p.first_of_target, INTERVAL i MONTH)
      )        AS period,
      'within'  AS period_type
    FROM
      params p,
      UNNEST(GENERATE_ARRAY(1,6)) AS i

    UNION ALL

    -- the bucket of "all prior" months, labeled as 7-months-ago
    SELECT
      FORMAT_DATE('%Y%m',
        DATE_SUB(p.first_of_target, INTERVAL 7 MONTH)
      )        AS period,
      'sum'     AS period_type
    FROM
      params p
  ),

  -- 5. Cross‐join repos × periods, then aggregate each bucket
  metrics_ts AS (
    SELECT
      cr.repo_name,
      pr.period,
      pr.period_type,

      -- age in days at end of each period:
      GREATEST(0, DATE_DIFF(
        CASE
          WHEN pr.period_type = 'within' THEN
            -- last day of that month
            DATE_SUB(
              DATE_ADD(PARSE_DATE('%Y%m', pr.period), INTERVAL 1 MONTH),
              INTERVAL 1 DAY
            )
          ELSE
            -- day before your 6-mo window begins
            DATE_SUB(
              DATE_SUB(p.first_of_target, INTERVAL 6 MONTH),
              INTERVAL 1 DAY
            )
        END,
        DATE(MIN(e.created_at)),
        DAY
      )) AS age_days,

      -- distinct users in the bucket
      COUNT(DISTINCT
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          ) THEN e.actor_id
        END
      ) AS users_involved,

      -- event‐type counts in the bucket
      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          )
          AND e.type = 'WatchEvent' THEN 1 ELSE 0
        END
      ) AS n_stars,

      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          )
          AND e.type = 'ForkEvent' THEN 1 ELSE 0
        END
      ) AS n_forks,

      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          )
          AND e.type = 'ReleaseEvent' THEN 1 ELSE 0
        END
      ) AS n_releases,

      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          )
          AND e.type = 'PullRequestEvent' THEN 1 ELSE 0
        END
      ) AS n_pulls,

      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          )
          AND e.type = 'IssuesEvent' THEN 1 ELSE 0
        END
      ) AS n_issues,

      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          )
          AND e.type IN ('IssueCommentEvent','PullRequestReviewCommentEvent')
        THEN 1 ELSE 0
        END
      ) AS n_comments,

      -- total events in the bucket
      SUM(
        CASE
          WHEN (
            (pr.period_type = 'within' AND e.ym = pr.period)
            OR
            (pr.period_type = 'sum'    AND e.ym < FORMAT_DATE('%Y%m',
               DATE_SUB(p.first_of_target, INTERVAL 6 MONTH)))
          ) THEN 1 ELSE 0
        END
      ) AS total_events

    FROM
      candid_repos cr
    CROSS JOIN
      params p
    CROSS JOIN
      periods pr
    LEFT JOIN
      events e
    ON
      e.repo_name = cr.repo_name
    GROUP BY
      cr.repo_name,
      pr.period,
      pr.period_type,
      p.first_of_target
  )

-- 6. Final: one row per repo×period, newest‐first
SELECT
  *
FROM
  metrics_ts
ORDER BY
  repo_name,
  period DESC;
"""


def format_bytes(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["bytes", "KB", "MB", "GB"]:
        if bytes_val < 1024 or unit == "GB":
            return (
                f"{bytes_val:.2f} {unit}" if unit != "bytes" else f"{bytes_val} {unit}"
            )
        bytes_val /= 1024
    return f"{bytes_val:.2f} GB"


def fetch_control_repos(target_month: str) -> None:
    """
    Fetch control repositories for a specific month and save to CSV.

    Args:
        target_month: Month in YYYYMM format (e.g., '202407')
    """
    output_path = (
        Path(__file__).parent.parent
        / "data"
        / f"control_repo_candidates_{target_month}.csv"
    )

    # Skip if file already exists
    if output_path.exists():
        logging.info(f"File already exists for {target_month}, skipping: {output_path}")
        return

    logging.info(f"Fetching control repositories for month {target_month}")

    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_month", "STRING", target_month)
        ]
    )

    # First run a dry run to estimate cost
    dry_run_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("target_month", "STRING", target_month)
        ],
        dry_run=True,
        use_query_cache=False,
    )
    query_job = client.query(CONTROL_REPO_QUERY, job_config=dry_run_config)
    bytes_processed = query_job.total_bytes_processed
    cost = bytes_processed / (1024**4) * 5.0

    logging.info(f"Estimated query size: {format_bytes(bytes_processed)}")
    logging.info(f"Estimated cost: ${cost:.4f} USD")

    logging.info(f"Executing query for {target_month}...")
    results = client.query(CONTROL_REPO_QUERY, job_config=job_config).result()
    df_results = results.to_dataframe()

    if df_results.empty:
        logging.warning(f"No control repositories found for {target_month}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    logging.info(f"Saved {len(df_results)} control repositories to {output_path}")


def main() -> None:
    """Main entry point for fetching control repositories."""
    parser = argparse.ArgumentParser(
        description="Fetch control GitHub repositories for specified months."
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=[
            "202408",
            "202409",
            "202410",
            "202411",
            "202412",
            "202501",
            "202502",
            "202503",
        ],
        help="List of months in YYYYMM format to fetch control repos for (default: 2024-08 to 2025-03)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Fetch control repositories for each specified month
    for month in args.months:
        fetch_control_repos(month)


if __name__ == "__main__":
    main()
