"""
Identify FALSE NEGATIVES from existing model predictions
This is much faster than re-running the model on all data
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os

print("="*80)
print("FALSE NEGATIVES ANALYSIS - Using Existing Predictions")
print("="*80)

# ============================================================
# APPROACH 1: Analyze documents with low confidence NON_HATE predictions
# ============================================================
print("\nAPPROACH 1: Low-confidence NON_HATE predictions in immigrant documents")
print("-"*80)

imm_df = pd.read_parquet(
    r'BERT_HessicherLandtag\Data\prep_v1\immigrant_docs_classified.parquet'
)

# Sort by prediction confidence (ascending)
# Documents predicted as NON_HATE with low confidence are suspicious
low_confidence_non_hate = imm_df[
    (imm_df['hate_label'] == 'NON_HATE')
].sort_values('hate_score')

print(f"\nTotal immigrant-mentioning documents: {len(imm_df)}")
print(f"Predicted as NON_HATE: {len(imm_df[imm_df['hate_label'] == 'NON_HATE'])}")
print(f"Predicted as HATE: {len(imm_df[imm_df['hate_label'] == 'HATE'])}")

print(f"\nDocuments classified as NON_HATE with LOWEST confidence:")
print(f"(These are most likely to be false negatives)\n")

output_dir = r'BERT_HessicherLandtag\Data\evaluation'
os.makedirs(output_dir, exist_ok=True)

# Show bottom 15 lowest confidence NON_HATE predictions
suspicious_docs = low_confidence_non_hate.head(15).copy()
print(f"{'Rank':<5} {'Confidence':<12} {'Text Preview'}")
print("-"*80)

for idx, (i, row) in enumerate(suspicious_docs.iterrows(), 1):
    confidence = row['hate_score']
    text_preview = row['text'][:80].replace('\n', ' ')
    print(f"{idx:<5} {confidence:<12.6f} {text_preview}...")

# Save suspicious (potentially false negative) documents
suspicious_path = os.path.join(output_dir, 'suspicious_non_hate_predictions.csv')
suspicious_docs[['doc_id', 'hate_label', 'hate_score', 'text']].to_csv(suspicious_path, index=False)
print(f"\n[OK] Saved {len(suspicious_docs)} suspicious documents to: {suspicious_path}")

# ============================================================
# APPROACH 2: Analyze manually labeled training data
# ============================================================
print("\n" + "="*80)
print("APPROACH 2: Analyzing Labeled Training Data")
print("-"*80)

labeled_df = pd.read_parquet(
    r'BERT_HessicherLandtag\Data\training\labeled_data.parquet'
)

print(f"\nTotal labeled documents: {len(labeled_df)}")
print(f"\nLabel distribution:")
print(labeled_df['label'].value_counts())

# Focus on HATE-labeled documents
hate_docs = labeled_df[labeled_df['label'] == 'HATE'].copy()
print(f"\nDocuments labeled as HATE: {len(hate_docs)}")
print(f"\nHATE-labeled documents (potential false negatives if model missed them):")
print("\nFirst 10 HATE examples:\n")
print(f"{'Index':<6} {'Text Preview'}")
print("-"*80)

for idx, (i, row) in enumerate(hate_docs.head(10).iterrows(), 1):
    text_preview = row['text'][:90].replace('\n', ' ')
    print(f"{idx:<6} {text_preview}...")

# Save all HATE-labeled documents for manual review
hate_path = os.path.join(output_dir, 'manually_labeled_hate_documents.csv')
hate_docs[['text', 'label']].to_csv(hate_path, index=False)
print(f"\n[OK] Saved all {len(hate_docs)} manually labeled HATE documents to: {hate_path}")

# ============================================================
# APPROACH 3: Cross-reference - Find documents that exist in both datasets
# ============================================================
print("\n" + "="*80)
print("APPROACH 3: Comparing Predictions with Labels")
print("-"*80)

# Extract text snippets for matching (since full texts might differ)
labeled_texts = set(labeled_df['text'].str[:100].values)
immigrant_texts = set(imm_df['text'].str[:100].values)

overlap = labeled_texts.intersection(immigrant_texts)
print(f"\nDocuments that appear in both datasets: {len(overlap)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("SUMMARY OF ANALYSIS")
print("="*80)

print(f"""
To identify FALSE NEGATIVES (documents that ARE hate speech but model missed):

1. SUSPICIOUS LOW-CONFIDENCE PREDICTIONS:
   - Found {len(low_confidence_non_hate)} documents predicted as NON_HATE
   - Bottom 15 saved to: {suspicious_path}
   - These have the lowest confidence and are most likely false negatives
   - Manual review recommended for documents with score < 0.98

2. MANUALLY LABELED HATE DOCUMENTS:
   - Total {len(hate_docs)} documents manually labeled as HATE
   - Saved to: {hate_path}
   - These are ground truth labels
   - If model predicted these as NON_HATE, they are false negatives

3. RECOMMENDATION:
   - Review the suspicious documents (lowest confidence NON_HATE predictions)
   - Compare with manually labeled HATE documents
   - Use these to understand model failures and improve performance

Output files created in: {output_dir}
""")

print("\nAnalysis complete!")
