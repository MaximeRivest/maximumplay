"""
Add Azure AI Inference and Anthropic Bedrock
"""
import pandas as pd
import requests
import json

# Packages to check
packages_to_add = {
    "Azure AI Inference": "azure-ai-inference",
    "Anthropic Bedrock": "anthropic-bedrock"
}

# Load current data
df = pd.read_csv("pypi_classified_final.csv")

for display_name, pkg_name in packages_to_add.items():
    try:
        # Get download stats
        r = requests.get(f"https://pypistats.org/api/packages/{pkg_name}/recent", timeout=10)
        if r.status_code == 200:
            downloads = r.json()["data"]["last_month"]
            print(f"{display_name} ({pkg_name}): {downloads:,} downloads")
            
            # Get package info
            r2 = requests.get(f"https://pypi.org/pypi/{pkg_name}/json", timeout=10)
            if r2.status_code == 200:
                info = r2.json().get("info", {})
                description = info.get("summary", "")[:100]
            else:
                description = ""
            
            # Add to dataframe if not exists
            if display_name not in df['project'].values:
                new_row = {
                    'project': display_name,
                    'pypi_package': pkg_name,
                    'downloads_last_month': downloads,
                    'description': description,
                    'tags': json.dumps(['model_provider']),
                    'primary_tag': 'model_provider',
                    'confidence': 0.9
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"  Added {display_name}")
        else:
            print(f"{display_name}: Not found (HTTP {r.status_code})")
    except Exception as e:
        print(f"{display_name}: Error - {e}")

# Sort by downloads
df = df.sort_values('downloads_last_month', ascending=False, na_position='last').reset_index(drop=True)

# Save
df.to_csv("pypi_classified_final.csv", index=False)

print("\n=== All Model Providers ===")
providers = df[df['primary_tag'] == 'model_provider'].sort_values('downloads_last_month', ascending=False)
for _, row in providers.iterrows():
    print(f"{row['project']:25} {row['downloads_last_month']:>12,.0f} downloads")