"""
False Positives finden: Fälle wo das neue Modell HATE vorhersagt,
aber das wahre Label NON_HATE ist.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

BASE_DIR   = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
MODEL_DIR  = BASE_DIR / "fine_tuned_model_combined"
EVAL_FILE  = BASE_DIR / "Data" / "evaluation" / "all_predictions_with_labels.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "evaluation"

BATCH_SIZE = 32

print("Lade Daten...")
df = pd.read_csv(EVAL_FILE)
print(f"  {len(df)} Einträge | Labels: {df['label'].value_counts().to_dict()}")

# Nur Einträge mit bekanntem Label
labeled = df[df['label'].isin(['HATE', 'NON_HATE'])].copy()
print(f"  Mit Label: {len(labeled)}")

print("\nLade Modell...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()
print(f"  Device: {device}")

print("\nKlassifiziere...")
texts = labeled['text'].tolist()
predictions, scores = [], []

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i:i + BATCH_SIZE]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        sc = probs[range(len(batch)), preds]
    predictions.extend(["HATE" if p.item() == 1 else "NON_HATE" for p in preds])
    scores.extend([round(s.item(), 4) for s in sc])
    if (i // BATCH_SIZE + 1) % 100 == 0:
        print(f"  {i + len(batch)}/{len(texts)}")

labeled['new_prediction'] = predictions
labeled['new_score'] = scores

# False Positives: Modell sagt HATE, wahres Label ist NON_HATE
fp = labeled[(labeled['new_prediction'] == 'HATE') & (labeled['label'] == 'NON_HATE')].copy()
# False Negatives: Modell sagt NON_HATE, wahres Label ist HATE
fn = labeled[(labeled['new_prediction'] == 'NON_HATE') & (labeled['label'] == 'HATE')].copy()

print(f"\nErgebnisse:")
print(f"  False Positives (HATE vorhergesagt, aber NON_HATE): {len(fp)}")
print(f"  False Negatives (NON_HATE vorhergesagt, aber HATE): {len(fn)}")

total_hate = (labeled['label'] == 'HATE').sum()
total_nonhate = (labeled['label'] == 'NON_HATE').sum()
print(f"\n  Precision HATE: {(len(labeled[(labeled['new_prediction']=='HATE') & (labeled['label']=='HATE')])) / max((labeled['new_prediction']=='HATE').sum(),1):.3f}")
print(f"  Recall HATE:    {(len(labeled[(labeled['new_prediction']=='HATE') & (labeled['label']=='HATE')])) / max(total_hate,1):.3f}")

# Speichern
fp_sorted = fp.sort_values('new_score', ascending=False)
fn_sorted = fn.sort_values('new_score', ascending=False)

fp_out = OUTPUT_DIR / "false_positives_combined_model.csv"
fn_out = OUTPUT_DIR / "false_negatives_combined_model.csv"
fp_sorted[['text', 'label', 'new_prediction', 'new_score']].to_csv(fp_out, index=False)
fn_sorted[['text', 'label', 'new_prediction', 'new_score']].to_csv(fn_out, index=False)

print(f"\n[OK] False Positives gespeichert: {fp_out}")
print(f"[OK] False Negatives gespeichert: {fn_out}")
print(f"\nTop 5 False Positives (höchste Confidence):")
print(fp_sorted[['text', 'new_score']].head().to_string())
