"""
Classify existing PyPI stats data using DSPy
"""
import json
import pandas as pd
import dspy
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-3.5-turbo", temperature=0.3), adapter=dspy.JSONAdapter())

# Classification categories
CLASSIFICATION_TAGS = [
    "model_provider",
    "hosting_provider", 
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
    """Classify an AI/LLM library based on its name and description."""
    library_name: str = dspy.InputField(desc="Name of the library")
    description: str = dspy.InputField(desc="Brief description of the library")
    available_tags: list[str] = dspy.InputField(desc="List of available classification tags")
    tags: list[str] = dspy.OutputField(desc="1-3 most applicable tags from available_tags")
    primary_tag: str = dspy.OutputField(desc="The single most applicable tag")
    confidence: float = dspy.OutputField(desc="Confidence score 0-1")

def classify_libraries():
    # Load existing data
    df = pd.read_csv("pypi_quick_stats.csv")
    df = df[df["downloads_last_month"].notna()]
    # Classify top 50 to include new libraries
    df = df.head(50)
    
    classifier = dspy.ChainOfThought(LibraryClassification)
    
    # Classify each library
    classifications = []
    print("Classifying libraries...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = classifier(
                library_name=row["project"],
                description=row["description"] or f"{row['project']} library",
                available_tags=CLASSIFICATION_TAGS
            )
            classifications.append({
                "project": row["project"],
                "tags": result.tags,
                "primary_tag": result.primary_tag,
                "confidence": result.confidence
            })
        except Exception as e:
            print(f"\nError classifying {row['project']}: {e}")
            classifications.append({
                "project": row["project"],
                "tags": [],
                "primary_tag": "unknown",
                "confidence": 0.0
            })
    
    # Merge with original data
    class_df = pd.DataFrame(classifications)
    df_final = df.merge(class_df, on="project")
    
    # Save results
    df_final.to_csv("pypi_classified.csv", index=False)
    print("\n✓ Saved to pypi_classified.csv")
    
    # Display results
    print("\n=== Top 20 Libraries with Classifications ===\n")
    display_df = df_final[["project", "downloads_last_month", "primary_tag", "confidence"]].head(20)
    print(tabulate(display_df, headers="keys", tablefmt="github", showindex=False, floatfmt=".0f"))
    
    # Tag distribution
    tag_counts = df_final["primary_tag"].value_counts()
    print("\n=== Primary Tag Distribution ===\n")
    for tag, count in tag_counts.items():
        print(f"{tag:25} {count:3d} libraries")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    tag_counts.plot(kind='bar')
    plt.title("AI/LLM Libraries by Primary Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Libraries")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("library_categories.png", dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to library_categories.png")

if __name__ == "__main__":
    classify_libraries()