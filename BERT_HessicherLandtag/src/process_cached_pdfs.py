import pymupdf
import torch
import re
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR   = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
CACHE_DIR  = BASE_DIR / "Data" / "protocols_pdf_cache"
OUTPUT_DIR = BASE_DIR / "Data" / "training"
MODEL_DIR  = BASE_DIR / "fine_tuned_model_cv" / "best_model"

SEGMENT_WORDS    = 350
OVERLAP          = 50
MIN_WORDS        = 30
CONFIDENCE       = 0.80
SAMPLE_PER_LABEL = 500   # reicht für Augmentation gegen Overfitting
CLASSIFY_SAMPLE  = 5000  # nur zufällige 5k Segmente klassifizieren
BATCH_SIZE       = 16


def clean(text):
    text = re.sub(r"(?m)^\s*\d{1,5}\s*$", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def segment(text):
    words = text.split()
    if len(words) < MIN_WORDS:
        return []
    segs, start = [], 0
    while start < len(words):
        end = min(start + SEGMENT_WORDS, len(words))
        s = " ".join(words[start:end])
        if len(s.split()) >= MIN_WORDS:
            segs.append(s)
        if end == len(words):
            break
        start += SEGMENT_WORDS - OVERLAP
    return segs


# 1. Text extrahieren
print("=" * 60)
print("SCHRITT 1: Text aus gecachten PDFs extrahieren")
print("=" * 60)
pdfs = sorted(CACHE_DIR.glob("*.pdf"))
print(f"  {len(pdfs)} PDFs gefunden")

all_segs = []
for i, pdf in enumerate(pdfs, 1):
    try:
        doc = pymupdf.open(str(pdf))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        text = clean(text)
        segs = segment(text)
        all_segs.extend(segs)
    except Exception:
        pass
    if i % 50 == 0 or i == len(pdfs):
        print(f"  {i}/{len(pdfs)} verarbeitet | Segmente: {len(all_segs)}")

print(f"\nSegmente gesamt: {len(all_segs)}")

# Zufälliges Sample für schnelle Klassifizierung
import random
random.seed(42)
if len(all_segs) > CLASSIFY_SAMPLE:
    all_segs = random.sample(all_segs, CLASSIFY_SAMPLE)
    print(f"Zufälliges Sample: {len(all_segs)} Segmente")

# 2. Modell laden
print("\n" + "=" * 60)
print("SCHRITT 2: Modell laden & klassifizieren")
print("=" * 60)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).to(device)
model.eval()
print("Modell geladen.")

results = []
for i in range(0, len(all_segs), BATCH_SIZE):
    batch = all_segs[i:i + BATCH_SIZE]
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        scores = probs[range(len(batch)), preds]
    for txt, pred, score in zip(batch, preds.cpu(), scores.cpu()):
        results.append({
            "text": txt,
            "label": "HATE" if pred.item() == 1 else "NON_HATE",
            "score": round(score.item(), 4)
        })
    if (i // BATCH_SIZE + 1) % 100 == 0:
        print(f"  {i + len(batch)}/{len(all_segs)} klassifiziert")

df = pd.DataFrame(results)
print(f"\nLabel-Verteilung (gesamt):\n{df['label'].value_counts()}")

# 3. Filtern & balancieren
print("\n" + "=" * 60)
print("SCHRITT 3: Filtern & balancieren")
print("=" * 60)
high = df[df["score"] >= CONFIDENCE]
print(f"High-confidence (>={CONFIDENCE}): {len(high)}")
print(high["label"].value_counts())

balanced = []
for label in high["label"].unique():
    sub = high[high["label"] == label]
    n = min(SAMPLE_PER_LABEL, len(sub))
    balanced.append(sub.sample(n=n, random_state=42))
    print(f"  {label}: {n} Samples")

result = pd.concat(balanced)[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

OUTPUT_DIR.mkdir(exist_ok=True)
out = OUTPUT_DIR / "labeled_data_hessen_protocols.csv"
result.to_csv(out, index=False)
print(f"\n[OK] Gespeichert: {out}")
print(f"     {len(result)} Samples total")
print(result["label"].value_counts())
