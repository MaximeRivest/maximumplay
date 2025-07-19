#!/usr/bin/env python3
"""
PyPIStats API Research: Understanding Download Data Breakdowns

This script demonstrates how to query the PyPIStats API to understand
what download data breakdowns are available.
"""

import json
import requests
from typing import Dict, Any
from datetime import datetime

def fetch_api_data(endpoint: str) -> Dict[str, Any]:
    """Fetch data from PyPIStats API endpoint."""
    base_url = "https://pypistats.org/api"
    url = f"{base_url}{endpoint}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {}

def analyze_package_downloads(package_name: str = "langchain"):
    """Analyze all available download data for a package."""
    
    print(f"\n{'='*60}")
    print(f"PyPIStats API Analysis for Package: {package_name}")
    print(f"{'='*60}\n")
    
    # 1. Recent Downloads
    print("1. RECENT DOWNLOADS")
    print("-" * 40)
    recent_data = fetch_api_data(f"/packages/{package_name}/recent")
    if recent_data:
        print(f"Response structure: {json.dumps(recent_data, indent=2)}")
        if 'data' in recent_data:
            data = recent_data['data']
            print(f"\nLast day: {data.get('last_day', 'N/A'):,}")
            print(f"Last week: {data.get('last_week', 'N/A'):,}")
            print(f"Last month: {data.get('last_month', 'N/A'):,}")
    
    # 2. Overall Downloads (with/without mirrors)
    print("\n\n2. OVERALL DOWNLOADS (Sample)")
    print("-" * 40)
    overall_data = fetch_api_data(f"/packages/{package_name}/overall")
    if overall_data and 'data' in overall_data:
        # Show first few entries to understand structure
        print("Categories available:")
        categories = set()
        for entry in overall_data['data'][:20]:
            categories.add(entry.get('category'))
        print(f"  - {', '.join(sorted(categories))}")
        
        # Show sample data
        print("\nSample entries:")
        for entry in overall_data['data'][:5]:
            print(f"  Date: {entry.get('date')}, "
                  f"Category: {entry.get('category')}, "
                  f"Downloads: {entry.get('downloads'):,}")
    
    # 3. Python Major Version
    print("\n\n3. PYTHON MAJOR VERSION BREAKDOWN (Sample)")
    print("-" * 40)
    python_major_data = fetch_api_data(f"/packages/{package_name}/python_major")
    if python_major_data and 'data' in python_major_data:
        # Show categories
        print("Python versions available:")
        versions = set()
        for entry in python_major_data['data'][:50]:
            versions.add(entry.get('category'))
        print(f"  - {', '.join(sorted(str(v) for v in versions))}")
        
        # Show latest day's breakdown
        latest_date = None
        for entry in reversed(python_major_data['data']):
            if latest_date is None:
                latest_date = entry.get('date')
            if entry.get('date') == latest_date:
                print(f"  Python {entry.get('category')}: {entry.get('downloads'):,} downloads")
    
    # 4. Python Minor Version
    print("\n\n4. PYTHON MINOR VERSION BREAKDOWN")
    print("-" * 40)
    python_minor_data = fetch_api_data(f"/packages/{package_name}/python_minor")
    if python_minor_data and 'data' in python_minor_data:
        # Show categories
        print("Python versions available:")
        versions = set()
        for entry in python_minor_data['data'][:200]:
            versions.add(entry.get('category'))
        print(f"  - {', '.join(sorted(str(v) for v in versions if v))}")
    
    # 5. System/OS Breakdown
    print("\n\n5. SYSTEM/OS BREAKDOWN (Sample)")
    print("-" * 40)
    system_data = fetch_api_data(f"/packages/{package_name}/system")
    if system_data and 'data' in system_data:
        # Show categories
        print("Operating systems available:")
        systems = set()
        for entry in system_data['data'][:50]:
            systems.add(entry.get('category'))
        print(f"  - {', '.join(sorted(str(s) for s in systems))}")
        
        # Show latest day's breakdown
        latest_date = None
        for entry in reversed(system_data['data']):
            if latest_date is None:
                latest_date = entry.get('date')
            if entry.get('date') == latest_date:
                print(f"  {entry.get('category') or 'Unknown'}: {entry.get('downloads'):,} downloads")

def check_for_installer_data():
    """Check if PyPIStats API provides installer type data."""
    print("\n\n" + "="*60)
    print("CHECKING FOR INSTALLER TYPE DATA")
    print("="*60)
    
    # Try some potential endpoints that might exist
    test_endpoints = [
        "/packages/pip/recent",
        "/packages/poetry/recent", 
        "/packages/uv/recent",
        "/packages/langchain/installer",  # Hypothetical
        "/packages/langchain/installers",  # Hypothetical
        "/packages/langchain/install_method",  # Hypothetical
    ]
    
    print("\nTesting potential installer-related endpoints:")
    for endpoint in test_endpoints:
        print(f"\nTrying: {endpoint}")
        data = fetch_api_data(endpoint)
        if data:
            print(f"  ✓ Success! Data keys: {list(data.keys())}")
        else:
            print(f"  ✗ No data or endpoint doesn't exist")

def main():
    """Main function to run all analyses."""
    # Analyze a popular package
    analyze_package_downloads("langchain")
    
    # Check for installer data
    check_for_installer_data()
    
    print("\n\n" + "="*60)
    print("SUMMARY: PyPIStats API Limitations")
    print("="*60)
    print("""
Based on the API exploration:

1. PyPIStats API provides these breakdowns:
   - Recent downloads (day/week/month totals)
   - Overall downloads (with_mirrors vs without_mirrors)
   - Python major version (2, 3, null)
   - Python minor version (2.7, 3.6, 3.7, etc.)
   - System/OS (Darwin, Linux, Windows, null)

2. PyPIStats API does NOT provide:
   - Installer type breakdown (pip, poetry, uv, etc.)
   - Direct install vs dependency install distinction
   - Download reason/context

3. For installer type data, you must use:
   - Google BigQuery dataset: bigquery-public-data.pypi.file_downloads
   - The 'details.installer.name' column in BigQuery
   - Tools like 'pypinfo' with --all flag

4. Mirror data notes:
   - 'with_mirrors' includes all downloads
   - 'without_mirrors' excludes known mirror services like bandersnatch
   - This is the closest to filtering "real" vs automated downloads
""")

if __name__ == "__main__":
    main()