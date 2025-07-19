"""
Add OpenRouter and fix Pydantic AI classification
"""
import pandas as pd
import requests

# First, let's check OpenRouter on PyPI
try:
    r = requests.get("https://pypistats.org/api/packages/openrouter/recent", timeout=10)
    if r.status_code == 200:
        openrouter_downloads = r.json()["data"]["last_month"]
        print(f"OpenRouter downloads: {openrouter_downloads:,}")
    else:
        print(f"OpenRouter not found: {r.status_code}")
        openrouter_downloads = None
except Exception as e:
    print(f"Error fetching OpenRouter: {e}")
    openrouter_downloads = None

# Load the current data
df = pd.read_csv("pypi_classified_final.csv")

# Fix Pydantic AI classification - it's now an agent framework
df.loc[df['project'] == 'pydantic‑ai', 'primary_tag'] = 'agentic_ai'

# Add OpenRouter if we found it
if openrouter_downloads:
    # Check if it already exists
    if 'OpenRouter' not in df['project'].values:
        new_row = {
            'project': 'OpenRouter',
            'pypi_package': 'openrouter',
            'downloads_last_month': openrouter_downloads,
            'description': 'Unified interface for LLM APIs',
            'tags': "['model_provider', 'ai_programming']",
            'primary_tag': 'model_provider',
            'confidence': 0.9
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print("Added OpenRouter to dataset")
    
# Sort by downloads
df = df.sort_values('downloads_last_month', ascending=False, na_position='last').reset_index(drop=True)

# Save updated data
df.to_csv("pypi_classified_final.csv", index=False)

print("\n=== Updated Classifications ===")
print(f"Pydantic AI is now classified as: {df[df['project'] == 'pydantic‑ai']['primary_tag'].values[0]}")

print("\n=== Agentic AI Libraries (Updated) ===")
agents = df[df['primary_tag'] == 'agentic_ai'].sort_values('downloads_last_month', ascending=False)
for _, row in agents.head(10).iterrows():
    print(f"{row['project']}: {row['downloads_last_month']:,.0f} downloads")

print("\n=== Model Providers (Checking for OpenRouter) ===")
providers = df[df['primary_tag'] == 'model_provider'].sort_values('downloads_last_month', ascending=False)
for _, row in providers.head(10).iterrows():
    print(f"{row['project']}: {row['downloads_last_month']:,.0f} downloads")