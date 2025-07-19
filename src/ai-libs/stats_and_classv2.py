"""
Enhanced PyPI Library Analyzer with DSPy Classification
Demonstrates Intent-Oriented Programming (IOP) for library categorization

Author: Assistant + You
Last updated: 2025-01-07
"""

from __future__ import annotations

import json
import time
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests
from requests.exceptions import RequestException
import dspy
from tqdm import tqdm
from tabulate import tabulate
import backoff

# ---------------------------------------------------------------------------
# 1. Catalogue — "display name" : exact PyPI package(s)
# ---------------------------------------------------------------------------
PACKAGES = {
    "Agno": ["agno"],
    "Agents SDK": ["openai-agents"],
    "AiSuite": ["aisuite"],
    "APE": ["ape-core"],
    "Arize AI": ["arize", "arize-phoenix"],
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
    "LM Studio": ["lmstudio"],
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
    "PROMST": [],
    "Promptex": ["promptex"],
    "PromptBench": ["promptbench"],
    "PromptFlow": ["promptflow"],
    "Promptim": [],
    "Replicate": ["replicate"],
    "Rivet": ["rivet"],
    "Semantic Kernel": ["semantic-kernel"],
    "Swarm (agent swarm)": ["swarms"],
    "Together AI": ["together"],
    "uAgents": ["uagents"],
    "Vertex AI Prompt Optimizer": [],
    # Convenience duplicates / aliases
    "pydantic‑ai": ["pydantic-ai"],
    "anthropic": ["anthropic"],
    # New additions
    "MAX (Modular)": ["max-engine", "modular", "mojo"],
    "vLLM": ["vllm"],
    "SGLang": ["sglang"],
}

# ---------------------------------------------------------------------------
# 2. DSPy Setup and Classification Signatures (Intent-Oriented Programming!)
# ---------------------------------------------------------------------------

# Configure DSPy with your preferred LLM and the JSONAdapter for robust parsing
# Using gpt-3.5-turbo for faster/cheaper classification
dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo", temperature=0.3), adapter=dspy.JSONAdapter())

# Define our classification categories
CLASSIFICATION_TAGS = [
    "model_provider",
    "hosting_provider",
    "model_training_lab",
    "ai_programming",
    "agentic_ai",
    "prompt_engineering",
    "structured_output",
    "evaluation_testing",
    "observability_monitoring",
    "retrieval_rag",
    "memory_state",
    "workflow_orchestration",
    "fine_tuning",
    "local_inference",
    "enterprise_platform",
]

class LibraryClassification(dspy.Signature):
    """Classify an AI/LLM library based on its description and README content."""
    library_name: str = dspy.InputField(desc="Name of the library")
    pypi_description: str = dspy.InputField(desc="Description from PyPI")
    github_readme: str = dspy.InputField(desc="README content from GitHub (may be truncated)")
    available_tags: List[str] = dspy.InputField(desc="List of available classification tags")
    tags: List[str] = dspy.OutputField(desc="List of applicable tags from available_tags")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")

class LibraryAnalyzer(dspy.Module):
    """DSPy Module for analyzing and classifying AI/LLM libraries."""
    def __init__(self):
        self.classify = dspy.ChainOfThought(LibraryClassification)
    def forward(self, library_name: str, pypi_description: str, github_readme: str) -> dspy.Prediction:
        # Truncate README to ~3000 chars (approx. 1000 tokens)
        if len(github_readme) > 3000:
            github_readme = github_readme[:3000] + "... [truncated]"
        return self.classify(
            library_name=library_name,
            pypi_description=pypi_description,
            github_readme=github_readme,
            available_tags=CLASSIFICATION_TAGS
        )

# ---------------------------------------------------------------------------
# 3. Data Fetching Functions (IOP-style robustness)
# ---------------------------------------------------------------------------

@backoff.on_exception(backoff.expo, RequestException, max_time=60)
def http_get(url, **kwargs):
    kwargs.setdefault("timeout", 10)
    return requests.get(url, **kwargs)

def fetch_last_month(pkg: str) -> tuple[Optional[int], Optional[int]]:
    """Return (total_downloads, without_mirrors_last_30days) or (None, None) if unavailable."""
    try:
        r = http_get(f"https://pypistats.org/api/packages/{pkg}/recent")
        if r.status_code == 200:
            total = r.json()["data"]["last_month"]
        else:
            print(f"\n  → {pkg}: HTTP {r.status_code} (recent)")
            return None, None

        r2 = http_get(f"https://pypistats.org/api/packages/{pkg}/overall")
        if r2.status_code == 200:
            data = r2.json()["data"]
            cutoff = date.today() - timedelta(days=30)
            without_mirrors = sum(
                row.get("downloads", 0)
                for row in data
                if "date" in row and date.fromisoformat(row["date"]) >= cutoff
            )
            return total, without_mirrors
        else:
            print(f"\n  → {pkg}: HTTP {r2.status_code} (overall)")
            return total, None
    except (RequestException, KeyError, json.JSONDecodeError, ValueError) as e:
        print(f"\n  → {pkg}: {type(e).__name__}")
    return None, None

def fetch_pypi_info(pkg: str) -> Dict[str, Any]:
    """Fetch package info from PyPI API."""
    try:
        r = http_get(f"https://pypi.org/pypi/{pkg}/json")
        if r.status_code == 200:
            data = r.json()
            info = data.get("info", {})
            return {
                "description": info.get("summary", ""),
                "home_page": info.get("home_page", ""),
                "project_urls": info.get("project_urls", {}),
            }
    except Exception as e:
        print(f"\n  → {pkg}: {type(e).__name__}")
    return {"description": "", "home_page": "", "project_urls": {}}

def extract_github_url(pypi_info: Dict[str, Any]) -> Optional[str]:
    """Extract GitHub URL from PyPI info."""
    urls = pypi_info.get("project_urls", {})
    if urls:
        for key, url in urls.items():
            if url and "github.com" in url.lower():
                return url
    home = pypi_info.get("home_page", "")
    if home and "github.com" in home.lower():
        return home
    return None

# Use env token for authenticated GitHub API if available
GITHUB_HEADERS = {
    "Accept": "application/vnd.github.raw",
    **({"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}
       if os.getenv("GITHUB_TOKEN") else {})
}

@backoff.on_exception(backoff.expo, RequestException, max_time=60)
def fetch_github_readme(github_url: str) -> str:
    """Fetch README from GitHub."""
    if not github_url:
        return ""
    match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
    if not match:
        return ""
    owner, repo = match.groups()
    repo = repo.rstrip('/')
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        r = http_get(api_url, headers=GITHUB_HEADERS)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print(f"  → GitHub fetch failed for {github_url}: {type(e).__name__}")
    return ""

@dataclass
class PackageMeta:
    name: str
    pypi_pkg: Optional[str]
    pypi_info: dict
    github_url: Optional[str]
    github_readme: str
    downloads_total: Optional[int]
    downloads_clean: Optional[int]

# ---------------------------------------------------------------------------
# 4. Main Analysis Pipeline
# ---------------------------------------------------------------------------

def hydrate_package(display_name: str, candidates: List[str]) -> Optional[PackageMeta]:
    """Fetch all remote/package meta needed for analysis."""
    if not candidates:
        return PackageMeta(
            name=display_name,
            pypi_pkg=None,
            pypi_info={},
            github_url=None,
            github_readme="",
            downloads_total=None,
            downloads_clean=None
        )
    for pkg in candidates:
        total, clean = fetch_last_month(pkg)
        if total is not None:
            pypi_info = fetch_pypi_info(pkg)
            github_url = extract_github_url(pypi_info)
            github_readme = fetch_github_readme(github_url) if github_url else ""
            return PackageMeta(
                name=display_name,
                pypi_pkg=pkg,
                pypi_info=pypi_info,
                github_url=github_url,
                github_readme=github_readme,
                downloads_total=total,
                downloads_clean=clean
            )
        time.sleep(1)  # avoid overloading PyPI API on misses
    return None  # All candidates failed

def analyze_libraries() -> pd.DataFrame:
    """Main function to analyze and classify libraries."""
    analyzer = LibraryAnalyzer()
    rows = []
    print("↳ Analyzing AI/LLM libraries...")
    for display_name, candidates in tqdm(PACKAGES.items(), unit="pkg"):
        time.sleep(0.5)  # gentle rate limit
        meta = hydrate_package(display_name, candidates)
        if meta is None:
            rows.append({
                "project": display_name,
                "pypi_package": " / ".join(candidates) if candidates else None,
                "downloads_last_month": float("nan"),
                "downloads_without_mirrors": float("nan"),
                "tags": [],
                "confidence": 0.0,
                "reasoning": "Package not found",
                "github_url": None
            })
            continue
        if not meta.pypi_pkg:
            rows.append({
                "project": display_name,
                "pypi_package": None,
                "downloads_last_month": float("nan"),
                "downloads_without_mirrors": float("nan"),
                "tags": [],
                "confidence": 0.0,
                "reasoning": "Not on PyPI",
                "github_url": None
            })
            continue
        try:
            classification = analyzer(
                library_name=meta.name,
                pypi_description=meta.pypi_info.get("description", ""),
                github_readme=meta.github_readme
            )
            # Ensure tags are list even if model returned a string
            tags = classification.tags
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            confidence = classification.confidence
            reasoning = classification.reasoning
        except Exception as e:
            print(f"\nClassification error for {meta.name}: {e}")
            tags = []
            confidence = 0.0
            reasoning = "Classification failed"
        rows.append({
            "project": meta.name,
            "pypi_package": meta.pypi_pkg,
            "downloads_last_month": meta.downloads_total,
            "downloads_without_mirrors": meta.downloads_clean,
            "tags": tags,
            "confidence": confidence,
            "reasoning": reasoning,
            "github_url": meta.github_url
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. Visualization and Display
# ---------------------------------------------------------------------------

def display_results(df: pd.DataFrame):
    """Display and visualize the results."""
    import matplotlib.pyplot as plt

    # Sort by downloads
    df = df.sort_values("downloads_last_month", ascending=False, na_position="last").reset_index(drop=True)
    # Save as JSON for tags field safety
    CACHE = Path("pypi_downloads_classified.json")
    df_out = df.copy()
    df_out["tags"] = df_out["tags"].apply(json.dumps)
    df_out.to_json(CACHE, orient="records", lines=True, force_ascii=False)
    print(f"\n✓ Saved classified data to {CACHE.resolve()}\n")
    # Display table with classifications
    display_df = df[["project", "pypi_package", "downloads_last_month", "tags"]].copy()
    display_df["tags"] = display_df["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    display_df.columns = ["Project", "PyPI Package", "Downloads (30d)", "Classification Tags"]
    print("\n=== Top 20 AI/LLM Libraries by Downloads ===\n")
    print(tabulate(display_df.head(20), headers="keys", tablefmt="github", showindex=False, floatfmt=".0f"))
    # Analyze tag distribution
    tag_counts = {}
    for tags in df["tags"]:
        if isinstance(tags, list):
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
    print("\n=== Tag Distribution ===\n")
    for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{tag:30} {count:3d} libraries")
    # Visualize top libraries
    TOP_N = 25
    top = df.head(TOP_N).iloc[::-1]
    plt.figure(figsize=(12, 0.4 * TOP_N + 2))
    bars = plt.barh(top["project"], top["downloads_last_month"])
    colors = {
        "model_provider": "#FF6B6B",
        "hosting_provider": "#4ECDC4",
        "ai_programming": "#45B7D1",
        "agentic_ai": "#96CEB4",
        "prompt_engineering": "#FECA57",
        "evaluation_testing": "#DDA0DD",
        "workflow_orchestration": "#98D8C8",
        "structured_output": "#ffb347",
        "retrieval_rag": "#5f9ea0",
        "fine_tuning": "#8d99ae",
        "memory_state": "#f7b801",
        "local_inference": "#9b59b6",
        "enterprise_platform": "#fd5f00",
        "observability_monitoring": "#008b8b",
        "model_training_lab": "#003366",
    }
    for i, (idx, row) in enumerate(top.iterrows()):
        if isinstance(row["tags"], list) and row["tags"]:
            primary_tag = row["tags"][0]
            color = colors.get(primary_tag, "#95A5A6")
            bars[i].set_color(color)
    plt.title(f"Top {TOP_N} AI/LLM Libraries – PyPI Downloads (Last 30 Days)")
    plt.xlabel("Downloads")
    plt.tight_layout()
    plt.show()
    # Create a second plot showing libraries by category
    fig, ax = plt.subplots(figsize=(10, 8))
    tag_lib_counts = {}
    for tag in CLASSIFICATION_TAGS:
        count = sum(1 for _, row in df.iterrows()
                   if isinstance(row["tags"], list) and tag in row["tags"]
                   and pd.notna(row["downloads_last_month"]))
        if count > 0:
            tag_lib_counts[tag] = count
    tags = list(tag_lib_counts.keys())
    counts = list(tag_lib_counts.values())
    plt.barh(tags, counts)
    plt.title("Number of Libraries by Category")
    plt.xlabel("Number of Libraries")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== AI/LLM Library Analyzer with DSPy Classification ===\n")
    print("This demonstrates Intent-Oriented Programming (IOP):")
    print("- We declare our classification intent via DSPy Signatures")
    print("- The system handles the implementation details")
    print("- Classification adapts to any LLM backend\n")
    df = analyze_libraries()
    display_results(df)
    print("\n✓ Analysis complete!")
    print("\nNote: This is a demonstration of IOP principles.")
    print("The classification quality depends on your configured LLM.")
    print("With DSPy, you could now optimize these classifications with labeled examples!")
