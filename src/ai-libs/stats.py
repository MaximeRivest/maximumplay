"""
Fetch PyPI download counts (last 30 days) for a catalogue of AI / LLM libraries
and visualise the results.

Author: <you>
Last updated: 2025‑07‑07
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException
from tqdm import tqdm
from tabulate import tabulate

# ---------------------------------------------------------------------------
# 1. Catalogue — “display name” : exact PyPI package(s)
#    (If a project publishes several related wheels, list alternates in order.)
# ---------------------------------------------------------------------------
PACKAGES = {
    # A–L
    "Agno": ["agno"],
    "Agents SDK": ["openai-agents"],
    "AiSuite": ["aisuite"],
    "APE": ["ape-core"],
    "Arize AI": ["arize", "arize-phoenix"],
    "AutoGen": ["autogen"],
    "BAML": ["baml"],
    "Claudette": ["claudette"],
    "Cohere": ["cohere"],
    "CrewAI": ["crewai"],
    "DeepEval": ["deepeval"],
    "DSPy": ["dspy"],
    "Ell": ["ell-ai"],
    "Flowise": ["flowise"],
    "Griptape": ["griptape"],
    "Groq": ["groq"],
    "Guardrails": ["guardrails-ai"],
    "Guidance": ["guidance", "llguidance"],
    "Haystack": ["haystack-ai", "farm-haystack"],
    "Instructor": ["instructor"],
    "LangChain": ["langchain"],
    "LangGraph": ["langgraph"],
    "LangMem": ["langmem"],
    "Langroid": ["langroid"],
    "Lilypad": ["python-lilypad"],
    "LiteLLM": ["litellm"],
    "LLM": ["llm"],
    "LM‑Format‑Enforcer": ["lm-format-enforcer"],
    "LM Studio": ["lmstudio"],
    "LlamaIndex": ["llama-index"],
    "MCP": ["mcp"],
    "Mem0": ["mem0ai"],
    "Mirascope": ["mirascope"],
    "Mistral": ["mistralai"],
    "Marvin": ["marvin"],
    # M–Z
    "Ollama": ["ollama"],
    "OpenAI": ["openai"],
    "OpenPrompt": ["openprompt"],
    "Outlines": ["outlines"],
    "Phidata": ["phidata"],
    "PROMST": [],                    # none on PyPI (yet)
    "Promptex": ["promptex"],
    "PromptBench": ["promptbench"],
    "PromptFlow": ["promptflow"],
    "Promptim": [],                  # none on PyPI (yet)
    "Replicate": ["replicate"],
    "Rivet": ["rivet"],
    "Semantic Kernel": ["semantic-kernel"],
    "Swarm (agent swarm)": ["swarms"],
    "Together AI": ["together"],
    "uAgents": ["uagents"],
    "Vertex AI Prompt Optimizer": [],  # lives in google‑cloud‑aiplatform
    # Convenience duplicates / aliases
    "pydantic‑ai": ["pydantic-ai"],
    "anthropic": ["anthropic"],
}

# ---------------------------------------------------------------------------
# 2. Helper – query PyPIStats ------------------------------------------------
# ---------------------------------------------------------------------------
API_RECENT_TMPL = "https://pypistats.org/api/packages/{pkg}/recent"
API_OVERALL_TMPL = "https://pypistats.org/api/packages/{pkg}/overall"

def fetch_last_month(pkg: str) -> tuple[int | None, int | None]:
    """Return (total_downloads, without_mirrors) or (None, None) if unavailable."""
    try:
        # Get recent data for total downloads
        r = requests.get(API_RECENT_TMPL.format(pkg=pkg), timeout=10)
        if r.status_code == 200:
            total = r.json()["data"]["last_month"]
            
            # Try to get without_mirrors data from overall endpoint
            r2 = requests.get(API_OVERALL_TMPL.format(pkg=pkg), timeout=10)
            if r2.status_code == 200:
                data = r2.json()["data"]
                # Sum last 30 days of without_mirrors
                without_mirrors = 0
                for entry in data[-30:]:  # Last 30 entries (days)
                    without_mirrors += entry.get("downloads", 0)
                return total, without_mirrors
            return total, None
        else:
            print(f"\n  → {pkg}: HTTP {r.status_code}")
    except (RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"\n  → {pkg}: {type(e).__name__}")
    return None, None


# ---------------------------------------------------------------------------
# 3. Crawl & assemble DataFrame ---------------------------------------------
# ---------------------------------------------------------------------------
rows = []

print("↳ Contacting PyPIStats.org …")
for display_name, candidates in tqdm(PACKAGES.items(), unit="pkg"):
    time.sleep(0.5)  # Rate limiting
    if not candidates:                   # project not on PyPI (yet)
        rows.append(
            dict(project=display_name, pypi_package=None, 
                 downloads_last_month=float("nan"),
                 downloads_without_mirrors=float("nan"))
        )
        continue

    for pkg in candidates:
        total, without_mirrors = fetch_last_month(pkg)
        if total is not None:
            rows.append(
                dict(project=display_name, 
                     pypi_package=pkg, 
                     downloads_last_month=total,
                     downloads_without_mirrors=without_mirrors)
            )
            break                       # stop at first successful alias
        time.sleep(1)                   # polite 1 req/s; adjust if needed
    else:                               # all aliases failed
        rows.append(
            dict(project=display_name, 
                 pypi_package=" / ".join(candidates),
                 downloads_last_month=float("nan"),
                 downloads_without_mirrors=float("nan"))
        )

df = (
    pd.DataFrame(rows)
      .sort_values("downloads_last_month", ascending=False, na_position="last")
      .reset_index(drop=True)
)

# Persist so you don’t hammer the API next time -----------------------------
CACHE = Path("pypi_downloads_detailed.csv")
df.to_csv(CACHE, index=False)
print(f"✓ Saved raw data to {CACHE.resolve()}\n")

# ---------------------------------------------------------------------------
# 4. Display table -----------------------------------------------------------
# Add a column for percentage of downloads without mirrors
df["mirror_pct"] = ((df["downloads_last_month"] - df["downloads_without_mirrors"]) / df["downloads_last_month"] * 100).round(1)
df["mirror_pct"] = df["mirror_pct"].fillna(0)

# Display subset of columns for readability
display_df = df[["project", "pypi_package", "downloads_last_month", "downloads_without_mirrors", "mirror_pct"]].copy()
display_df.columns = ["Project", "PyPI Package", "Total Downloads", "Without Mirrors", "Mirror %"]
print(tabulate(display_df.head(30), headers="keys", tablefmt="github", showindex=False, floatfmt=".0f"))

# ---------------------------------------------------------------------------
# 5. Visualise — horizontal bar chart ---------------------------------------
TOP_N = 25                             # top‑25 for readability
top = df.head(TOP_N).iloc[::-1]        # flip for horizontal bars

plt.figure(figsize=(10, 0.4 * TOP_N + 2))
plt.barh(top["project"], top["downloads_last_month"])
plt.title(f"Top {TOP_N} libraries – PyPI downloads in the last 30 days")
plt.xlabel("Downloads")
plt.tight_layout()
plt.show()
