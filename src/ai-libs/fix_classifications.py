"""
Fix misclassifications in the data
"""
import pandas as pd

# Load the classified data
df = pd.read_csv("pypi_classified.csv")

# Manual corrections based on what these libraries actually do
corrections = {
    # Model providers (API clients)
    "OpenAI": "model_provider",
    "anthropic": "model_provider", 
    "Cohere": "model_provider",
    "Mistral": "model_provider",
    "Groq": "model_provider",
    "Replicate": "model_provider",
    "Together AI": "model_provider",
    
    # Local inference engines
    "Ollama": "local_inference",
    "vLLM": "local_inference",
    "SGLang": "local_inference",
    "MAX (Modular)": "local_inference",
    "LM Studio": "local_inference",
    
    # AI Programming frameworks
    "LangChain": "ai_programming",
    "LangGraph": "workflow_orchestration",
    "LlamaIndex": "retrieval_rag",
    "DSPy": "ai_programming",
    "Semantic Kernel": "ai_programming",
    "Haystack": "retrieval_rag",
    "LiteLLM": "ai_programming",  # It's a unified interface, not a provider
    "AiSuite": "ai_programming",
    "Guidance": "ai_programming",
    "Griptape": "ai_programming",
    "Mirascope": "ai_programming",
    "Ell": "ai_programming",
    
    # Structured output
    "Instructor": "structured_output",
    "Outlines": "structured_output",
    "LM‑Format‑Enforcer": "structured_output",
    "pydantic‑ai": "structured_output",  # It's about structured output with Pydantic
    "BAML": "structured_output",
    
    # Agentic AI
    "CrewAI": "agentic_ai",
    "AutoGen": "agentic_ai",
    "Agents SDK": "agentic_ai",
    "Agno": "agentic_ai",
    "Swarm (agent swarm)": "agentic_ai",
    "Langroid": "agentic_ai",
    "uAgents": "agentic_ai",
    "Phidata": "agentic_ai",
    
    # Evaluation & Testing
    "DeepEval": "evaluation_testing",
    "Guardrails": "evaluation_testing",
    "PromptBench": "evaluation_testing",
    
    # Observability & Monitoring
    "Arize AI": "observability_monitoring",
    
    # Memory & State
    "Mem0": "memory_state",
    "LangMem": "memory_state",
    
    # Prompt Engineering
    "PromptFlow": "prompt_engineering",
    "OpenPrompt": "prompt_engineering",
    "Promptex": "prompt_engineering",
    "Claudette": "prompt_engineering",  # It's a Claude wrapper focused on prompting
    
    # Other
    "MCP": "workflow_orchestration",  # Model Context Protocol is about orchestration
    "LLM": "ai_programming",  # Simon Willison's CLI tool
    "Marvin": "ai_programming",
    "Flowise": "workflow_orchestration",  # Visual workflow builder
}

# Apply corrections
for project, correct_tag in corrections.items():
    df.loc[df['project'] == project, 'primary_tag'] = correct_tag

# Save corrected data
df.to_csv("pypi_classified_corrected.csv", index=False)

# Show category counts
print("\n=== Corrected Category Distribution ===")
print(df['primary_tag'].value_counts().to_string())

# Show top libraries by category
print("\n=== Top Libraries by Category ===")
for category in df['primary_tag'].unique():
    if pd.notna(category):
        libs = df[df['primary_tag'] == category].head(5)
        print(f"\n{category}:")
        for _, row in libs.iterrows():
            print(f"  - {row['project']}: {row['downloads_last_month']:,.0f} downloads")