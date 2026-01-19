import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
INPUT_PATH = r'C:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\hate_speech_against_immigrants.csv'
OUTPUT_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\visualizations'
FIGSIZE = (12, 8)
DPI = 300

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = FIGSIZE
plt.rcParams['font.size'] = 10



print("="*80)
print("CLASSIFICATION RESULTS VISUALIZATION")
print("="*80)

# Load the classified data
print(f"\nLoading classified data from: {INPUT_PATH}")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} classified documents")
print(f"Columns: {df.columns.tolist()}")

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================================
# EXTRACT PARTY MENTIONS
# ============================================================================
print("\nExtracting party mentions from documents...")

# Define parties and their colors
parties = {
    'CDU': '#000000',      # Black
    'SPD': '#E3000F',      # Red
    'GRÜNE': '#64A12D',    # Green
    'FDP': '#FFED00',      # Yellow
    'LINKE': '#BE3075',    # Magenta
    'AfD': '#009EE0'       # Blue
}

# Create columns for each party mention
for party in parties.keys():
    if party == 'GRÜNE':
        df[party] = df['text'].str.contains(r'GRÜNE|GRÜNEN|Bündnis\s*90', case=False, regex=True, na=False)
    else:
        df[party] = df['text'].str.contains(party, case=False, na=False)

# ============================================================================
# 1. PARTY MENTION COUNTS
# ============================================================================
print("\n[1/2] Creating party mention counts plot...")

fig, ax = plt.subplots(figsize=(10, 6))
party_counts = {party: df[party].sum() for party in parties.keys()}
party_names = list(party_counts.keys())
counts = list(party_counts.values())
colors = [parties[p] for p in party_names]

bars = ax.bar(party_names, counts, color=colors, alpha=0.8, edgecolor='black')
ax.set_xlabel('Political Party')
ax.set_ylabel('Number of Documents')
ax.set_title('Documents with Hate Speech Against Immigrants by Party Mention', fontsize=14, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hate_speech_by_party_counts.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. HATE SCORE DISTRIBUTION BY PARTY (Box Plot)
# ============================================================================
print("\n[2/2] Creating hate score distribution by party plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for box plot
party_data = []
party_labels = []
party_colors = []

for party in parties.keys():
    scores = df[df[party]]['hate_score'].values
    if len(scores) > 0:
        party_data.append(scores)
        party_labels.append(f'{party}\n(n={len(scores)})')
        party_colors.append(parties[party])

bp = ax.boxplot(party_data, labels=party_labels, patch_artist=True)

for patch, color in zip(bp['boxes'], party_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Political Party')
ax.set_ylabel('Hate Score')
ax.set_title('Distribution of Hate Scores by Party Mention', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/hate_score_distribution_by_party.png', dpi=DPI, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print(f"Output saved to: {OUTPUT_DIR}/")
print("  - hate_speech_by_party_counts.png")
print("  - hate_score_distribution_by_party.png")
print("="*80)