"""
Script to identify documents with FALSE NEGATIVES
False Negatives = Documents where:
- True label: HATE (actually is hate speech)
- Predicted label: NON_HATE (model failed to detect it)
"""

import torch
from transformers import pipeline
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

# ============================================================
# 1. SETUP
# ============================================================
print("="*80)
print("IDENTIFYING FALSE NEGATIVES IN HATE SPEECH DETECTION")
print("="*80)

# Check GPU availability
if torch.cuda.is_available():
    device = 0
    print("\n[GPU] CUDA GPU available")
else:
    device = -1
    print("\n[CPU] Using CPU (slower)")

# Initialize the model pipeline
model_path = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\fine_tuned_model_cv\best_model"
print(f"\nLoading model: {model_path}")

pipe = pipeline(
    "text-classification",
    model=model_path,
    device=device,
    batch_size=8,
    truncation=True
)
print("[OK] Model loaded successfully")

# ============================================================
# 2. LOAD LABELED DATA
# ============================================================
labeled_data_path = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\labeled_data.parquet"
print(f"\nLoading labeled data: {labeled_data_path}")

df = pd.read_parquet(labeled_data_path)
print(f"[OK] Loaded {len(df)} labeled documents")
print(f"\nTrue label distribution:")
print(df['label'].value_counts())

# ============================================================
# 3. RUN PREDICTIONS
# ============================================================
print("\nRunning model predictions on all labeled documents...")
texts = df['text'].tolist()

predictions = []
for i in tqdm(range(0, len(texts), 8), desc="Classifying documents"):
    batch = texts[i:i+8]
    batch_results = pipe(batch)
    predictions.extend(batch_results)

# Extract predicted labels and scores
df['predicted_label'] = [p['label'] for p in predictions]
df['predicted_score'] = [p['score'] for p in predictions]

print("[OK] Predictions complete")

# ============================================================
# 4. IDENTIFY FALSE NEGATIVES
# ============================================================
print("\n" + "="*80)
print("FALSE NEGATIVES ANALYSIS")
print("="*80)

false_negatives = df[(df['label'] == 'HATE') & (df['predicted_label'] == 'NON_HATE')].copy()
false_negatives = false_negatives.sort_values('predicted_score')

print(f"\nFound {len(false_negatives)} FALSE NEGATIVES")
print(f"(Documents that ARE hate speech but model classified as NON_HATE)")

if len(false_negatives) > 0:
    print(f"\nLowest confidence false negatives (most concerning):")
    print("\n" + "-"*80)
    for idx, (i, row) in enumerate(false_negatives.head(10).iterrows(), 1):
        print(f"\n[{idx}] Document Index: {i}")
        print(f"    True Label: {row['label']}")
        print(f"    Predicted: {row['predicted_label']} (confidence: {row['predicted_score']:.4f})")
        print(f"    Text preview: {row['text'][:200]}...")
else:
    print("\n[OK] No false negatives found! Model correctly identified all HATE documents.")

# ============================================================
# 5. CONFUSION MATRIX & METRICS
# ============================================================
print("\n" + "="*80)
print("OVERALL MODEL PERFORMANCE")
print("="*80)

# Map labels to numeric values
label_map = {'NON_HATE': 0, 'HATE': 1}
y_true = df['label'].map(label_map)
y_pred = df['predicted_label'].map(label_map)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print("                 Predicted NON_HATE | Predicted HATE")
print(f"True NON_HATE:        {cm[0, 0]:>4}          |      {cm[0, 1]:>4}")
print(f"True HATE:            {cm[1, 0]:>4}          |      {cm[1, 1]:>4}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['NON_HATE', 'HATE']))

# ============================================================
# 6. SAVE RESULTS
# ============================================================
output_dir = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\evaluation"

import os
os.makedirs(output_dir, exist_ok=True)

# Save all results
all_results_path = os.path.join(output_dir, "all_predictions_with_labels.csv")
df.to_csv(all_results_path, index=False)
print(f"\n[OK] Saved all predictions: {all_results_path}")

# Save only false negatives
if len(false_negatives) > 0:
    false_negatives_path = os.path.join(output_dir, "false_negatives.csv")
    false_negatives.to_csv(false_negatives_path, index=False)
    print(f"[OK] Saved false negatives: {false_negatives_path}")
    
    false_negatives_summary = false_negatives[['predicted_score', 'text']].copy()
    false_negatives_summary['text'] = false_negatives_summary['text'].str[:300]
    summary_path = os.path.join(output_dir, "false_negatives_summary.csv")
    false_negatives_summary.to_csv(summary_path, index=False)
    print(f"[OK] Saved false negatives summary: {summary_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
