#!/usr/bin/env python3
"""
BigQuery Analysis for PyPI Download Data with Installer Type Breakdowns

This script demonstrates how to query Google BigQuery to get installer type
breakdowns that aren't available through the PyPIStats API.

Note: Requires Google Cloud credentials and the google-cloud-bigquery package.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta

# Example BigQuery queries for installer type analysis

INSTALLER_BREAKDOWN_QUERY = """
#standardSQL
SELECT 
  details.installer.name AS installer_name,
  COUNT(*) AS download_count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM `bigquery-public-data.pypi.file_downloads`
WHERE file.project = @package_name
  AND DATE(timestamp) BETWEEN @start_date AND @end_date
GROUP BY details.installer.name
ORDER BY download_count DESC
"""

INSTALLER_TIME_SERIES_QUERY = """
#standardSQL
SELECT 
  DATE(timestamp) AS download_date,
  details.installer.name AS installer_name,
  COUNT(*) AS download_count
FROM `bigquery-public-data.pypi.file_downloads`
WHERE file.project = @package_name
  AND DATE(timestamp) BETWEEN @start_date AND @end_date
  AND details.installer.name IN ('pip', 'poetry', 'uv', 'pipenv', 'conda')
GROUP BY download_date, installer_name
ORDER BY download_date DESC, download_count DESC
"""

# Query to identify potential direct vs dependency installs
# This is heuristic-based since there's no direct flag
INSTALL_CONTEXT_QUERY = """
#standardSQL
WITH hourly_downloads AS (
  SELECT 
    DATETIME_TRUNC(timestamp, HOUR) AS download_hour,
    details.installer.name AS installer,
    file.project,
    COUNT(*) AS downloads_in_hour
  FROM `bigquery-public-data.pypi.file_downloads`
  WHERE file.project = @package_name
    AND DATE(timestamp) BETWEEN @start_date AND @end_date
  GROUP BY download_hour, installer, project
),
-- Calculate download patterns that might indicate dependency installs
download_patterns AS (
  SELECT 
    download_hour,
    installer,
    downloads_in_hour,
    -- High volume in short time might indicate CI/CD or dependency resolution
    CASE 
      WHEN downloads_in_hour > 1000 THEN 'likely_automated'
      WHEN downloads_in_hour > 100 THEN 'possibly_automated'
      ELSE 'likely_human'
    END AS download_pattern
  FROM hourly_downloads
)
SELECT 
  installer,
  download_pattern,
  COUNT(*) AS pattern_count,
  SUM(downloads_in_hour) AS total_downloads
FROM download_patterns
GROUP BY installer, download_pattern
ORDER BY installer, download_pattern
"""

# Query to see what other packages are commonly downloaded together
# This can help identify dependency relationships
PACKAGE_CORRELATION_QUERY = """
#standardSQL
WITH user_sessions AS (
  -- Group downloads by IP and time window to approximate "sessions"
  SELECT 
    details.installer.name AS installer,
    file.project,
    FARM_FINGERPRINT(CONCAT(
      country_code, 
      DATE(timestamp), 
      EXTRACT(HOUR FROM timestamp)
    )) AS session_id
  FROM `bigquery-public-data.pypi.file_downloads`
  WHERE DATE(timestamp) = @target_date
    AND details.installer.name = 'pip'
),
package_pairs AS (
  SELECT 
    a.project AS package_a,
    b.project AS package_b,
    COUNT(DISTINCT a.session_id) AS co_occurrence_count
  FROM user_sessions a
  JOIN user_sessions b 
    ON a.session_id = b.session_id 
    AND a.project < b.project
  WHERE a.project = @package_name OR b.project = @package_name
  GROUP BY package_a, package_b
  HAVING co_occurrence_count > 10
)
SELECT * FROM package_pairs
ORDER BY co_occurrence_count DESC
LIMIT 20
"""

def print_query_example(title: str, query: str, description: str):
    """Print a formatted query example."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"\nDescription: {description}")
    print(f"\nQuery:")
    print("-" * 40)
    print(query)
    print("-" * 40)

def generate_pypinfo_commands():
    """Generate example pypinfo commands for installer analysis."""
    print("\n\n" + "="*60)
    print("PYPINFO COMMAND EXAMPLES")
    print("="*60)
    
    commands = [
        {
            "description": "Get installer breakdown for a package (last 30 days)",
            "command": "pypinfo --all langchain installer"
        },
        {
            "description": "Get installer breakdown for specific date range",
            "command": "pypinfo --all --start-date 2025-06-01 --end-date 2025-06-30 langchain installer"
        },
        {
            "description": "Get top packages by installer type",
            "command": "pypinfo --all --limit 20 '' project installer | grep -E '(pip|poetry|uv)'"
        },
        {
            "description": "Get Python version breakdown by installer",
            "command": "pypinfo --all langchain installer pyversion"
        },
        {
            "description": "Get OS breakdown by installer", 
            "command": "pypinfo --all langchain installer system"
        },
        {
            "description": "Compare pip vs all installers for a package",
            "command": '''echo "Pip only:" && pypinfo langchain && echo -e "\\nAll installers:" && pypinfo --all langchain'''
        }
    ]
    
    for cmd in commands:
        print(f"\n{cmd['description']}:")
        print(f"$ {cmd['command']}")

def main():
    """Main function to demonstrate BigQuery queries."""
    
    print("BigQuery Analysis for PyPI Installer Type Data")
    print("=" * 60)
    
    # Show installer breakdown query
    print_query_example(
        "1. INSTALLER TYPE BREAKDOWN",
        INSTALLER_BREAKDOWN_QUERY,
        "Shows the breakdown of downloads by installer type (pip, poetry, uv, etc.)"
    )
    
    # Show time series query
    print_query_example(
        "2. INSTALLER USAGE OVER TIME",
        INSTALLER_TIME_SERIES_QUERY,
        "Shows how different installers' usage changes over time"
    )
    
    # Show install context query
    print_query_example(
        "3. INSTALL CONTEXT ANALYSIS (Direct vs Dependency)",
        INSTALL_CONTEXT_QUERY,
        "Attempts to identify automated/dependency installs vs direct human installs based on download patterns"
    )
    
    # Show package correlation query
    print_query_example(
        "4. PACKAGE CO-OCCURRENCE ANALYSIS",
        PACKAGE_CORRELATION_QUERY,
        "Shows which packages are commonly downloaded together, helping identify dependency relationships"
    )
    
    # Generate pypinfo commands
    generate_pypinfo_commands()
    
    # Summary
    print("\n\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    print("""
1. Direct vs Dependency Install Detection:
   - BigQuery does NOT have a direct flag for this
   - Must use heuristics like:
     * Download volume patterns
     * Time clustering
     * Package co-occurrence
     * Known CI/CD IP ranges

2. Available Installer Data:
   - details.installer.name: pip, poetry, uv, pipenv, conda, etc.
   - details.installer.version: Version of the installer
   - Can filter/group by these fields

3. Cost Considerations:
   - Filtering by installer costs ~25% more in BigQuery
   - Use date ranges to limit data scanned
   - 1TB free tier per month

4. Recommended Approach:
   - Use pypinfo with --all flag for quick analysis
   - Use BigQuery directly for complex queries
   - Combine with other signals (time patterns, volume) to infer install context
""")

if __name__ == "__main__":
    main()