import pandas as pd

# Read the v2 data (without Redis)
df = pd.read_csv("pypi_downloads_last_monthv2.csv")

# Remove Redis if it's still there
df = df[df['project'] != 'Redis (client)']

# Save as a new file
df.to_csv("pypi_downloads_ai_only.csv", index=False)

print(f"Saved {len(df)} AI/LLM libraries to pypi_downloads_ai_only.csv")
print("\nTop 10 AI/LLM libraries:")
print(df.head(10).to_string(index=False))