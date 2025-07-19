
"""
Enhanced PyPI Library Analyzer with DSPy Classification
Demonstrates Intent-Oriented Programming for library categorization

Author: Assistant + You
Last updated: 2025-01-07
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.exceptions import RequestException
from tqdm import tqdm
from tabulate import tabulate
import dspy

# ---------------------------------------------------------------------------
# 1. Catalogue — "display name" : exact PyPI package(s)
# ---------------------------------------------------------------------------
PACKAGES = {
    # A–L
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
}

# ---------------------------------------------------------------------------
# 2. DSPy Setup and Classification Signatures (Intent-Oriented Programming!)
# ---------------------------------------------------------------------------

# Configure DSPy with your preferred LLM
dspy.configure(lm=dspy.LM("openai/gpt-4.1", temperature=0.3))

# Define our classification categories
CLASSIFICATION_TAGS = [
    "model_provider",           # From an LLM/model provider (OpenAI, Anthropic, etc.)
    "hosting_provider",         # From a model hosting service (Replicate, Together, etc.)
    "model_training_lab",       # From a research lab that trains models
    "ai_programming",           # For AI/LLM programming and development
    "agentic_ai",              # For building AI agents and multi-agent systems
    "prompt_engineering",       # Focused on prompt optimization/engineering
    "structured_output",        # For enforcing structured outputs from LLMs
    "evaluation_testing",       # For evaluating and testing AI systems
    "observability_monitoring", # For monitoring and observability of AI apps
    "retrieval_rag",           # For retrieval-augmented generation
    "memory_state",            # For memory and state management in AI apps
    "workflow_orchestration",   # For orchestrating AI workflows
    "fine_tuning",             # For fine-tuning models
    "local_inference",         # For running models locally
    "enterprise_platform",      # Enterprise AI platform
]

class LibraryClassification(dspy.Signature):
    """Classify an AI/LLM library based on its description and README content.

    Analyze the library's purpose, features, and origin to assign appropriate tags.
    Be precise and only assign tags that clearly apply."""

    library_name: str = dspy.InputField(desc="Name of the library")
    pypi_description: str = dspy.InputField(desc="Description from PyPI")
    github_readme: str = dspy.InputField(desc="README content from GitHub (may be truncated)")
    available_tags: List[str] = dspy.InputField(desc="List of available classification tags")

    tags: List[str] = dspy.OutputField(desc="List of applicable tags from available_tags")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")
    reasoning: str = dspy.OutputField(desc="Brief explanation for the classification")

class LibraryAnalyzer(dspy.Module):
    """DSPy Module for analyzing and classifying AI/LLM libraries"""

    def __init__(self):
        self.classify = dspy.ChainOfThought(LibraryClassification)

    def forward(self, library_name: str, pypi_description: str, github_readme: str) -> dspy.Prediction:
        # Truncate README if too long (to stay within token limits)
        if len(github_readme) > 3000:
            github_readme = github_readme[:3000] + "... [truncated]"

        return self.classify(
            library_name=library_name,
            pypi_description=pypi_description,
            github_readme=github_readme,
            available_tags=CLASSIFICATION_TAGS
        )

# ---------------------------------------------------------------------------
# 3. Data Fetching Functions
# ---------------------------------------------------------------------------

API_RECENT_TMPL = "https://pypistats.org/api/packages/{pkg}/recent"
API_OVERALL_TMPL = "https://pypistats.org/api/packages/{pkg}/overall"
PYPI_API_TMPL = "https://pypi.org/pypi/{pkg}/json"

def fetch_last_month(pkg: str) -> tuple[int | None, int | None]:
    """Return (total_downloads, without_mirrors) or (None, None) if unavailable."""
    try:
        r = requests.get(API_RECENT_TMPL.format(pkg=pkg), timeout=10)
        if r.status_code == 200:
            total = r.json()["data"]["last_month"]

            r2 = requests.get(API_OVERALL_TMPL.format(pkg=pkg), timeout=10)
            if r2.status_code == 200:
                data = r2.json()["data"]
                without_mirrors = 0
                for entry in data[-30:]:
                    without_mirrors += entry.get("downloads", 0)
                return total, without_mirrors
            return total, None
        else:
            print(f"\n  → {pkg}: HTTP {r.status_code}")
    except (RequestException, KeyError, json.JSONDecodeError) as e:
        print(f"\n  → {pkg}: {type(e).__name__}")
    return None, None

def fetch_pypi_info(pkg: str) -> Dict[str, Any]:
    """Fetch package info from PyPI API"""
    try:
        r = requests.get(PYPI_API_TMPL.format(pkg=pkg), timeout=10)
        if r.status_code == 200:
            data = r.json()
            info = data.get("info", {})
            return {
                "description": info.get("summary", ""),
                "home_page": info.get("home_page", ""),
                "project_urls": info.get("project_urls", {}),
            }
    except:
        pass
    return {"description": "", "home_page": "", "project_urls": {}}

def extract_github_url(pypi_info: Dict[str, Any]) -> Optional[str]:
    """Extract GitHub URL from PyPI info"""
    # Check project URLs first
    urls = pypi_info.get("project_urls", {})
    for key, url in urls.items():
        if "github.com" in url.lower():
            return url

    # Check home page
    home = pypi_info.get("home_page", "")
    if "github.com" in home.lower():
        return home

    return None

def fetch_github_readme(github_url: str) -> str:
    """Fetch README from GitHub"""
    if not github_url:
        return ""

    # Extract owner/repo from URL
    match = re.search(r'github\.com/([^/]+)/([^/]+)', github_url)
    if not match:
        return ""

    owner, repo = match.groups()
    repo = repo.rstrip('/')

    # Try to fetch README via GitHub API
    api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.raw"}

    try:
        r = requests.get(api_url, headers=headers, timeout=10)
        if r.status_code == 200:
            return r.text
    except:
        pass

    return ""

# ---------------------------------------------------------------------------
# 4. Main Analysis Pipeline
# ---------------------------------------------------------------------------

def analyze_libraries():
    """Main function to analyze and classify libraries"""

    analyzer = LibraryAnalyzer()
    rows = []

    print("↳ Analyzing AI/LLM libraries...")

    for display_name, candidates in tqdm(PACKAGES.items(), unit="pkg"):
        time.sleep(0.5)  # Rate limiting

        if not candidates:
            rows.append({
                "project": display_name,
                "pypi_package": None,
                "downloads_last_month": float("nan"),
                "downloads_without_mirrors": float("nan"),
                "tags": [],
                "confidence": 0.0,
                "reasoning": "Not on PyPI"
            })
            continue

        # Find the first successful candidate
        for pkg in candidates:
            total, without_mirrors = fetch_last_month(pkg)
            if total is not None:
                # Fetch additional info for classification
                pypi_info = fetch_pypi_info(pkg)
                github_url = extract_github_url(pypi_info)
                github_readme = fetch_github_readme(github_url) if github_url else ""

                # Use DSPy to classify the library
                try:
                    classification = analyzer(
                        library_name=display_name,
                        pypi_description=pypi_info.get("description", ""),
                        github_readme=github_readme
                    )

                    tags = classification.tags
                    confidence = classification.confidence
                    reasoning = classification.reasoning
                except Exception as e:
                    print(f"\nClassification error for {display_name}: {e}")
                    tags = []
                    confidence = 0.0
                    reasoning = "Classification failed"

                rows.append({
                    "project": display_name,
                    "pypi_package": pkg,
                    "downloads_last_month": total,
                    "downloads_without_mirrors": without_mirrors,
                    "tags": tags,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "github_url": github_url
                })
                break
            time.sleep(1)
        else:
            rows.append({
                "project": display_name,
                "pypi_package": " / ".join(candidates),
                "downloads_last_month": float("nan"),
                "downloads_without_mirrors": float("nan"),
                "tags": [],
                "confidence": 0.0,
                "reasoning": "Package not found"
            })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. Visualization and Display
# ---------------------------------------------------------------------------

def display_results(df: pd.DataFrame):
    """Display and visualize the results"""

    # Sort by downloads
    df = df.sort_values("downloads_last_month", ascending=False, na_position="last").reset_index(drop=True)

    # Save full results
    CACHE = Path("pypi_downloads_classified.csv")
    df.to_csv(CACHE, index=False)
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

    # Color bars by primary category
    colors = {
        "model_provider": "#FF6B6B",
        "hosting_provider": "#4ECDC4",
        "ai_programming": "#45B7D1",
        "agentic_ai": "#96CEB4",
        "prompt_engineering": "#FECA57",
        "evaluation_testing": "#DDA0DD",
        "workflow_orchestration": "#98D8C8",
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

    # Count libraries per tag (only for libraries with downloads)
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
