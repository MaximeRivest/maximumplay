"""
Quick stats fetcher without classification
"""
import json
import time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from tabulate import tabulate

# Same package list
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
    "pydantic‑ai": ["pydantic-ai"],
    "anthropic": ["anthropic"],
    # New additions
    "MAX (Modular)": ["max-engine", "modular", "mojo"],
    "vLLM": ["vllm"],
    "SGLang": ["sglang"],
}

def fetch_last_month(pkg: str) -> int | None:
    """Fetch last month downloads."""
    try:
        r = requests.get(f"https://pypistats.org/api/packages/{pkg}/recent", timeout=10)
        if r.status_code == 200:
            return r.json()["data"]["last_month"]
    except Exception as e:
        print(f"\n  → {pkg}: {type(e).__name__}")
    return None

def fetch_pypi_info(pkg: str) -> dict:
    """Fetch basic PyPI info."""
    try:
        r = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=10)
        if r.status_code == 200:
            info = r.json().get("info", {})
            return {
                "description": info.get("summary", ""),
                "home_page": info.get("home_page", ""),
                "project_urls": info.get("project_urls", {}),
            }
    except Exception:
        pass
    return {"description": "", "home_page": "", "project_urls": {}}

def main():
    rows = []
    print("↳ Fetching PyPI stats...")
    
    for display_name, candidates in tqdm(PACKAGES.items(), unit="pkg"):
        time.sleep(0.3)
        
        if not candidates:
            rows.append({
                "project": display_name,
                "pypi_package": None,
                "downloads_last_month": float("nan"),
                "description": ""
            })
            continue
            
        for pkg in candidates:
            downloads = fetch_last_month(pkg)
            if downloads is not None:
                info = fetch_pypi_info(pkg)
                rows.append({
                    "project": display_name,
                    "pypi_package": pkg,
                    "downloads_last_month": downloads,
                    "description": (info.get("description") or "")[:100]
                })
                break
            time.sleep(0.5)
        else:
            rows.append({
                "project": display_name,
                "pypi_package": " / ".join(candidates),
                "downloads_last_month": float("nan"),
                "description": ""
            })
    
    df = pd.DataFrame(rows).sort_values("downloads_last_month", ascending=False, na_position="last")
    
    # Save
    df.to_csv("pypi_quick_stats.csv", index=False)
    print(f"\n✓ Saved to pypi_quick_stats.csv")
    
    # Display top 20
    display_df = df[["project", "pypi_package", "downloads_last_month", "description"]].head(20)
    print("\n=== Top 20 AI/LLM Libraries ===\n")
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False, floatfmt=".0f"))

if __name__ == "__main__":
    main()