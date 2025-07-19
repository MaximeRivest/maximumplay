"""
Create a comprehensive summary of the AI/LLM ecosystem
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load final data
df = pd.read_csv("pypi_classified_final.csv")
df = df[df['downloads_last_month'].notna()]

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('AI/LLM Ecosystem Analysis - PyPI Downloads (Last 30 Days)', fontsize=20, fontweight='bold')

# 1. Top Libraries by Category (Stacked Bar)
categories = ['model_provider', 'ai_programming', 'agentic_ai', 'local_inference', 
              'workflow_orchestration', 'structured_output', 'retrieval_rag']
colors = {
    'model_provider': '#e74c3c',
    'ai_programming': '#3498db', 
    'local_inference': '#2ecc71',
    'agentic_ai': '#9b59b6',
    'workflow_orchestration': '#1abc9c',
    'structured_output': '#f39c12',
    'retrieval_rag': '#34495e'
}

# Get top 3 from each category
data_for_stack = []
for cat in categories:
    cat_df = df[df['primary_tag'] == cat].nlargest(3, 'downloads_last_month')
    for _, row in cat_df.iterrows():
        data_for_stack.append({
            'library': row['project'][:15] + '...' if len(row['project']) > 15 else row['project'],
            'category': cat,
            'downloads': row['downloads_last_month'] / 1e6
        })

stack_df = pd.DataFrame(data_for_stack)
pivot_df = stack_df.pivot(index='library', columns='category', values='downloads').fillna(0)

pivot_df.plot(kind='barh', stacked=True, ax=ax1, color=[colors.get(c, '#95a5a6') for c in pivot_df.columns])
ax1.set_title('Top Libraries by Category', fontsize=14, fontweight='bold')
ax1.set_xlabel('Downloads (Millions)')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. Category Distribution
cat_summary = df.groupby('primary_tag').agg({
    'downloads_last_month': 'sum',
    'project': 'count'
}).sort_values('downloads_last_month', ascending=False)

cat_summary['downloads_millions'] = cat_summary['downloads_last_month'] / 1e6
top_cats = cat_summary.head(7)

ax2.barh(top_cats.index, top_cats['downloads_millions'], 
         color=[colors.get(cat, '#95a5a6') for cat in top_cats.index])
ax2.set_xlabel('Total Downloads (Millions)')
ax2.set_title('Downloads by Category', fontsize=14, fontweight='bold')

# Add count labels
for i, (idx, row) in enumerate(top_cats.iterrows()):
    ax2.text(row['downloads_millions'] + 1, i, f"{row['project']} libs", 
             va='center', fontsize=10)

# 3. Agent Frameworks Comparison
agents_df = df[df['primary_tag'] == 'agentic_ai'].sort_values('downloads_last_month', ascending=False).head(8)
ax3.bar(range(len(agents_df)), agents_df['downloads_last_month'] / 1e3, color='#9b59b6')
ax3.set_xticks(range(len(agents_df)))
ax3.set_xticklabels(agents_df['project'].values, rotation=45, ha='right')
ax3.set_ylabel('Downloads (Thousands)')
ax3.set_title('Agent Frameworks Adoption', fontsize=14, fontweight='bold')

# 4. Inference Solutions
inference_cats = ['local_inference', 'model_provider']
inference_df = df[df['primary_tag'].isin(inference_cats)].copy()

# Separate local vs cloud
local_df = inference_df[inference_df['primary_tag'] == 'local_inference'].head(5)
cloud_df = inference_df[inference_df['primary_tag'] == 'model_provider'].head(5)

y_pos = list(range(len(local_df))) + list(range(len(local_df) + 1, len(local_df) + len(cloud_df) + 1))
libraries = list(local_df['project']) + list(cloud_df['project'])
downloads = list(local_df['downloads_last_month'] / 1e6) + list(cloud_df['downloads_last_month'] / 1e6)
colors_list = ['#2ecc71'] * len(local_df) + ['#e74c3c'] * len(cloud_df)

ax4.barh(y_pos, downloads, color=colors_list)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(libraries)
ax4.set_xlabel('Downloads (Millions)')
ax4.set_title('Local vs Cloud Inference', fontsize=14, fontweight='bold')

# Add divider line
ax4.axhline(y=len(local_df) + 0.5, color='gray', linestyle='--', alpha=0.5)
ax4.text(0.5, len(local_df) + 0.5, 'LOCAL', transform=ax4.get_yaxis_transform(), 
         va='bottom', ha='center', fontweight='bold', color='#2ecc71')
ax4.text(0.5, len(local_df) + 0.5, 'CLOUD', transform=ax4.get_yaxis_transform(), 
         va='top', ha='center', fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig('ai_ecosystem_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Print key insights
print("=== Key Insights ===")
print(f"\n1. Pydantic AI ({df[df['project'] == 'pydantic‑ai']['downloads_last_month'].values[0]:,.0f} downloads) is now classified as an Agent Framework")
print(f"\n2. Azure AI Inference ({df[df['project'] == 'Azure AI Inference']['downloads_last_month'].values[0]:,.0f} downloads) is a major cloud provider")
print(f"\n3. Local inference is dominated by Ollama (2.7M) and vLLM (2.1M)")
print(f"\n4. SGLang (265K downloads) is growing in the high-performance serving space")
print(f"\n5. OpenRouter exists but has modest adoption (7.8K downloads)")

print("\n✓ Created comprehensive summary: ai_ecosystem_summary.png")