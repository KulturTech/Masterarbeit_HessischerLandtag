import pandas as pd
from pathlib import Path

"""
This script creates a labeled training dataset from the classified results.

There are two approaches:
1. Use high-confidence predictions as "silver labels" (automated)
2. Manually label a sample of documents (manual)

This script implements approach #1 - using high-confidence predictions.
"""

# Configuration
INPUT_PATH = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_classified.parquet'
OUTPUT_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training'
OUTPUT_FILE = 'labeled_data.parquet'

# Confidence threshold - only use predictions above this threshold
CONFIDENCE_THRESHOLD = 0.7  # Use 0.7 to include more HATE samples (adjust as needed)
SAMPLE_SIZE_PER_LABEL = 500  # Number of examples per label

print("="*80)
print("CREATING LABELED TRAINING DATASET")
print("="*80)

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load classified data
print(f"\nLoading classified data from: {INPUT_PATH}")
df = pd.read_parquet(INPUT_PATH)
print(f"Total documents: {len(df)}")

# Show label distribution
print(f"\nOriginal label distribution:")
print(df['label'].value_counts())
print(f"\nOriginal confidence statistics:")
print(df['score'].describe())

# Filter for high-confidence predictions
print(f"\nFiltering for high-confidence predictions (score >= {CONFIDENCE_THRESHOLD})...")
high_conf_df = df[df['score'] >= CONFIDENCE_THRESHOLD].copy()
print(f"High-confidence documents: {len(high_conf_df)}")

# Show distribution after filtering
print(f"\nHigh-confidence label distribution:")
print(high_conf_df['label'].value_counts())

# Sample balanced dataset (equal number per label, if possible)
print(f"\nCreating balanced training set ({SAMPLE_SIZE_PER_LABEL} samples per label)...")
labeled_datasets = []

for label in high_conf_df['label'].unique():
    label_df = high_conf_df[high_conf_df['label'] == label]

    # Sample up to SAMPLE_SIZE_PER_LABEL examples
    n_samples = min(SAMPLE_SIZE_PER_LABEL, len(label_df))
    sampled = label_df.sample(n=n_samples, random_state=42)
    labeled_datasets.append(sampled)

    print(f"  {label}: {n_samples} samples")

# Combine all samples
labeled_df = pd.concat(labeled_datasets, ignore_index=True)

# Keep only necessary columns for training
labeled_df = labeled_df[['text', 'label']].copy()

# Shuffle the dataset
labeled_df = labeled_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal training dataset:")
print(f"  Total samples: {len(labeled_df)}")
print(f"  Label distribution:")
print(labeled_df['label'].value_counts())

# Save the labeled dataset
output_path = f"{OUTPUT_DIR}/{OUTPUT_FILE}"
labeled_df.to_parquet(output_path)
print(f"\n[OK] Labeled dataset saved to: {output_path}")

# Also save as CSV for manual review/editing
csv_path = f"{OUTPUT_DIR}/labeled_data.csv"
labeled_df.to_csv(csv_path, index=False)
print(f"[OK] Also saved as CSV for review: {csv_path}")

print("\n" + "="*80)
print("DATASET CREATION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review the dataset (check the CSV file)")
print("2. Optionally edit labels manually if needed")
print("3. Run fine-tuning: python src/fine_tune.py")
print("\nNotes:")
print(f"- Using confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"- Higher threshold = more reliable labels but fewer samples")
print(f"- Lower threshold = more samples but potentially noisier labels")
print("- You can adjust CONFIDENCE_THRESHOLD and SAMPLE_SIZE_PER_LABEL in this script")
