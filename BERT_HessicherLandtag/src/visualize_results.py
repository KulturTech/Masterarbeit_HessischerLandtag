import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
INPUT_PATH = 'BERT_HessicherLandtag/Data/prep_v1/all_docs_classified.parquet'
OUTPUT_DIR = 'BERT_HessicherLandtag/Data/visualizations'
FIGSIZE = (12, 8)
DPI = 300

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = FIGSIZE
plt.rcParams['font.size'] = 10

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("="*80)
print("CLASSIFICATION RESULTS VISUALIZATION")
print("="*80)

# Load the classified data
print(f"\nLoading classified data from: {INPUT_PATH}")
df = pd.read_parquet(INPUT_PATH)
print(f"Loaded {len(df)} classified documents")
print(f"Columns: {df.columns.tolist()}")

# ============================================================================
# 1. LABEL DISTRIBUTION
# ============================================================================
print("\n[1/8] Creating label distribution plot...")

fig, ax = plt.subplots(figsize=(10, 6))
label_counts = df['label'].value_counts()

# Bar plot
bars = ax.bar(range(len(label_counts)), label_counts.values, color='steelblue', alpha=0.8)
ax.set_xticks(range(len(label_counts)))
ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
ax.set_ylabel('Number of Documents')
ax.set_xlabel('Label')
ax.set_title('Distribution of Classification Labels', fontsize=14, fontweight='bold')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/1_label_distribution.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. CONFIDENCE SCORE DISTRIBUTION
# ============================================================================
print("[2/8] Creating confidence score distribution...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Overall distribution
axes[0].hist(df['score'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(df['score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["score"].mean():.3f}')
axes[0].axvline(df['score'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df["score"].median():.3f}')
axes[0].set_xlabel('Confidence Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Overall Confidence Score Distribution', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Distribution by label
for label in df['label'].unique():
    label_scores = df[df['label'] == label]['score']
    axes[1].hist(label_scores, bins=30, alpha=0.5, label=label, edgecolor='black')

axes[1].set_xlabel('Confidence Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Confidence Score Distribution by Label', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/2_confidence_scores.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. BOX PLOT OF CONFIDENCE SCORES BY LABEL
# ============================================================================
print("[3/8] Creating confidence score box plot...")

fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='score', by='label', ax=ax, patch_artist=True)
ax.set_xlabel('Label')
ax.set_ylabel('Confidence Score')
ax.set_title('Confidence Score Distribution by Label', fontweight='bold')
plt.suptitle('')  # Remove default title
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/3_score_boxplot.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. LOW CONFIDENCE PREDICTIONS
# ============================================================================
print("[4/8] Analyzing low confidence predictions...")

# Define confidence thresholds
low_confidence_threshold = 0.6
medium_confidence_threshold = 0.8

df['confidence_level'] = pd.cut(df['score'],
                                bins=[0, low_confidence_threshold, medium_confidence_threshold, 1.0],
                                labels=['Low (<0.6)', 'Medium (0.6-0.8)', 'High (>0.8)'])

fig, ax = plt.subplots(figsize=(10, 6))
confidence_counts = df['confidence_level'].value_counts().sort_index()
bars = ax.bar(range(len(confidence_counts)), confidence_counts.values,
              color=['#d32f2f', '#ff9800', '#388e3c'], alpha=0.8)

ax.set_xticks(range(len(confidence_counts)))
ax.set_xticklabels(confidence_counts.index)
ax.set_ylabel('Number of Documents')
ax.set_xlabel('Confidence Level')
ax.set_title('Documents by Confidence Level', fontsize=14, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_confidence_levels.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. HEATMAP: CONFIDENCE BY LABEL
# ============================================================================
print("[5/8] Creating confidence heatmap...")

fig, ax = plt.subplots(figsize=(10, 8))
pivot_data = df.groupby(['label', 'confidence_level']).size().unstack(fill_value=0)
sns.heatmap(pivot_data, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Number of Predictions by Label and Confidence Level', fontsize=14, fontweight='bold')
ax.set_xlabel('Confidence Level')
ax.set_ylabel('Label')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/5_confidence_heatmap.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. TEXT LENGTH ANALYSIS
# ============================================================================
print("[6/8] Analyzing text length distribution...")

df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Character length distribution
axes[0, 0].hist(df['text_length'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Text Length (characters)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Text Length Distribution', fontweight='bold')
axes[0, 0].axvline(df['text_length'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["text_length"].mean():.0f}')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Word count distribution
axes[0, 1].hist(df['word_count'], bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Word Count')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Word Count Distribution', fontweight='bold')
axes[0, 1].axvline(df['word_count'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["word_count"].mean():.0f}')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Text length by label
df.boxplot(column='text_length', by='label', ax=axes[1, 0], patch_artist=True)
axes[1, 0].set_xlabel('Label')
axes[1, 0].set_ylabel('Text Length (characters)')
axes[1, 0].set_title('Text Length by Label', fontweight='bold')
axes[1, 0].get_figure().suptitle('')

# Word count by label
df.boxplot(column='word_count', by='label', ax=axes[1, 1], patch_artist=True)
axes[1, 1].set_xlabel('Label')
axes[1, 1].set_ylabel('Word Count')
axes[1, 1].set_title('Word Count by Label', fontweight='bold')
axes[1, 1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/6_text_length_analysis.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. CORRELATION: TEXT LENGTH vs CONFIDENCE
# ============================================================================
print("[7/8] Creating text length vs confidence scatter plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Character length vs confidence
axes[0].scatter(df['text_length'], df['score'], alpha=0.3, s=10)
axes[0].set_xlabel('Text Length (characters)')
axes[0].set_ylabel('Confidence Score')
axes[0].set_title('Text Length vs Confidence Score', fontweight='bold')
axes[0].grid(alpha=0.3)

# Add correlation
corr = df['text_length'].corr(df['score'])
axes[0].text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=axes[0].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Word count vs confidence
axes[1].scatter(df['word_count'], df['score'], alpha=0.3, s=10, color='coral')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Confidence Score')
axes[1].set_title('Word Count vs Confidence Score', fontweight='bold')
axes[1].grid(alpha=0.3)

# Add correlation
corr = df['word_count'].corr(df['score'])
axes[1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=axes[1].transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/7_length_vs_confidence.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. SUMMARY STATISTICS TABLE
# ============================================================================
print("[8/8] Creating summary statistics...")

fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# Create summary statistics
summary_stats = []
for label in sorted(df['label'].unique()):
    label_df = df[df['label'] == label]
    stats = {
        'Label': label,
        'Count': len(label_df),
        'Percentage': f"{len(label_df)/len(df)*100:.1f}%",
        'Mean Score': f"{label_df['score'].mean():.3f}",
        'Median Score': f"{label_df['score'].median():.3f}",
        'Std Score': f"{label_df['score'].std():.3f}",
        'Mean Text Length': f"{label_df['text_length'].mean():.0f}",
        'Mean Word Count': f"{label_df['word_count'].mean():.0f}"
    }
    summary_stats.append(stats)

# Add overall stats
overall_stats = {
    'Label': 'OVERALL',
    'Count': len(df),
    'Percentage': '100.0%',
    'Mean Score': f"{df['score'].mean():.3f}",
    'Median Score': f"{df['score'].median():.3f}",
    'Std Score': f"{df['score'].std():.3f}",
    'Mean Text Length': f"{df['text_length'].mean():.0f}",
    'Mean Word Count': f"{df['word_count'].mean():.0f}"
}
summary_stats.append(overall_stats)

# Create table
summary_df = pd.DataFrame(summary_stats)
table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style the table
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight overall row
for i in range(len(summary_df.columns)):
    table[(len(summary_stats), i)].set_facecolor('#FFC000')
    table[(len(summary_stats), i)].set_text_props(weight='bold')

ax.set_title('Classification Summary Statistics', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/8_summary_statistics.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ============================================================================
# SAVE DETAILED STATISTICS TO CSV
# ============================================================================
print("\nSaving detailed statistics to CSV...")

# Save low confidence predictions
low_conf_df = df[df['score'] < low_confidence_threshold][['doc_id', 'text', 'label', 'score']].copy()
low_conf_df = low_conf_df.sort_values('score')
low_conf_df.to_csv(f'{OUTPUT_DIR}/low_confidence_predictions.csv', index=False)
print(f"  - Low confidence predictions: {len(low_conf_df)} documents saved")

# Save summary statistics
summary_df.to_csv(f'{OUTPUT_DIR}/summary_statistics.csv', index=False)
print(f"  - Summary statistics saved")

# Save per-label statistics
label_stats = df.groupby('label').agg({
    'score': ['count', 'mean', 'median', 'std', 'min', 'max'],
    'text_length': ['mean', 'median', 'std'],
    'word_count': ['mean', 'median', 'std']
}).round(3)
label_stats.to_csv(f'{OUTPUT_DIR}/label_statistics.csv')
print(f"  - Per-label statistics saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION COMPLETE!")
print("="*80)
print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
print("\nGenerated files:")
print("  1. 1_label_distribution.png - Bar chart of label counts")
print("  2. 2_confidence_scores.png - Histogram of confidence scores")
print("  3. 3_score_boxplot.png - Box plot of scores by label")
print("  4. 4_confidence_levels.png - Documents by confidence level")
print("  5. 5_confidence_heatmap.png - Heatmap of confidence by label")
print("  6. 6_text_length_analysis.png - Text length distributions")
print("  7. 7_length_vs_confidence.png - Scatter plots")
print("  8. 8_summary_statistics.png - Summary table")
print("\nCSV files:")
print("  - low_confidence_predictions.csv - Predictions with score < 0.6")
print("  - summary_statistics.csv - Overall summary")
print("  - label_statistics.csv - Detailed per-label statistics")

print(f"\n{'='*80}")
print("KEY INSIGHTS:")
print("="*80)
print(f"Total Documents: {len(df)}")
print(f"Labels: {df['label'].nunique()}")
print(f"\nMost Common Label: {label_counts.index[0]} ({label_counts.values[0]} docs, {label_counts.values[0]/len(df)*100:.1f}%)")
print(f"Average Confidence: {df['score'].mean():.3f}")
print(f"Median Confidence: {df['score'].median():.3f}")
print(f"\nLow Confidence (<{low_confidence_threshold}): {len(low_conf_df)} documents ({len(low_conf_df)/len(df)*100:.1f}%)")
print(f"Medium Confidence ({low_confidence_threshold}-{medium_confidence_threshold}): {len(df[(df['score'] >= low_confidence_threshold) & (df['score'] < medium_confidence_threshold)])} documents")
print(f"High Confidence (>{medium_confidence_threshold}): {len(df[df['score'] >= medium_confidence_threshold])} documents")
