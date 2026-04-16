"""
Annotationsstichprobe erstellen
================================
1. Lädt alle Parlamentssegmente aus all_docs_clean.parquet
2. Filtert TOC/Kopfzeilen heraus → nur echte Redeabschnitte
3. Extrahiert Absätze mit Migrations-Keywords (+ Kontext)
4. Bewertet alle Absätze mit dem vortrainierten dehatebert-mono-german
5. Zieht eine stratifizierte Stichprobe:
     - 150 Texte mit P(HATE) > 0.15  (hohes HATE-Potenzial)
     -  50 Texte zufällig             (Repräsentation der Basis)
   → 200 Texte insgesamt für die manuelle Annotation
6. Speichert annotation_sample.csv → Data/annotation/

Ausführung:
  python src/prepare_annotation_sample.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Pfade & Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR       = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
DATA_PATH      = BASE_DIR / "Data" / "prep_v1" / "all_docs_clean.parquet"
OUT_DIR        = BASE_DIR / "Data" / "annotation"
OUT_CSV        = OUT_DIR / "annotation_sample.csv"

PRETRAINED_MODEL = "Hate-speech-CNERG/dehatebert-mono-german"
BATCH_SIZE       = 16
MAX_LENGTH       = 256

N_HIGH_HATE  = 150   # Texte mit hohem P(HATE)
N_RANDOM     = 50    # Zufällige Texte (Basis-Repräsentation)
HATE_THRESH  = 0.15  # Schwelle für "hohes HATE-Potenzial"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Migration-Keywords (wie in detect_immigrant_hate_speech.py)
# ---------------------------------------------------------------------------
_KEYWORD_PATTERNS = [
    r'\b(immigrant\w*|immigranten|einwanderer\w*|einwanderung\w*|zuwander\w*|zuwanderung\w*)\b',
    r'\b(migrant\w*|migranten|migration\w*)\b',
    r'\b(flüchtling\w*|asyl\w*|asylbewerber\w*|asylsuchende\w*)\b',
    r'\b(ausländer\w*|ausländisch\w*)\b',
    r'\b(geflüchtete\w*|schutzsuchende\w*)\b',
    r'\b(integration\w*)\b',
    r'\b(abschiebung\w*|abgeschoben\w*|rückführung\w*)\b',
    r'\b(grenzsicherung\w*|grenzschutz\w*|grenzkontrollen\w*)\b',
    r'\b(asylpolitik\w*|migrationspolitik\w*|ausländerpolitik\w*)\b',
]
_MIGRATION_RE = re.compile('|'.join(_KEYWORD_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def is_toc(text: str) -> bool:
    """Gibt True zurück, wenn der Text ein Inhaltsverzeichnis/Kopfzeile ist."""
    if not isinstance(text, str):
        return True
    t = text.strip()
    lines = [l for l in t.splitlines() if l.strip()]
    if not lines or len(t) < 100:
        return True
    dotted = sum(1 for l in lines if re.search(r'\.{4,}\s*\d*\s*$', l))
    return (dotted / len(lines)) > 0.3


def extract_migration_paragraphs(doc_id: str, text: str) -> list[dict]:
    """Teilt einen Seitentext in Absätze und extrahiert jene mit Migrations-Keywords."""
    paragraphs = re.split(r'\n{2,}', text)
    results = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 80 or not _MIGRATION_RE.search(para):
            continue
        ctx_before = paragraphs[i - 1].strip() if i > 0 else ''
        ctx_after  = paragraphs[i + 1].strip() if i < len(paragraphs) - 1 else ''
        chunk = '\n\n'.join(p for p in [ctx_before, para, ctx_after] if len(p) > 40)
        if len(chunk) > 1200:
            chunk = chunk[:1200]
        results.append({'doc_id': doc_id, 'text': chunk})
    return results


def score_with_pretrained(texts: list[str], tokenizer, model, device) -> np.ndarray:
    """Gibt P(HATE) für jeden Text zurück (shape: n,)."""
    # Finde HATE-Index aus dem Modell-Config
    id2label = model.config.id2label
    hate_idx = next((k for k, v in id2label.items() if v == 'HATE'), 1)

    all_probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Scoring", unit="batch"):
        batch = texts[i: i + BATCH_SIZE]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        all_probs.append(probs[:, hate_idx])
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Daten laden
    print(f"Lade Daten: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Gesamt: {len(df)} Segmente")

    # 2. TOC/Kopfzeilen entfernen
    speech = df[~df['text'].apply(is_toc)].copy()
    print(f"  Nach TOC-Filter: {len(speech)} Redeabschnitte")

    # 3. Absätze mit Migrations-Keywords extrahieren
    print("Extrahiere Migrations-Absätze...")
    rows = []
    for _, row in speech.iterrows():
        rows.extend(extract_migration_paragraphs(row['doc_id'], row['text']))
    chunks = pd.DataFrame(rows).drop_duplicates(subset='text').reset_index(drop=True)
    print(f"  {len(chunks)} einzigartige Absätze extrahiert")

    # 4. Vortrainiertes Modell laden & Texte bewerten
    print(f"\nLade vortrainiertes Modell: {PRETRAINED_MODEL}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL).to(device)
    model.eval()

    print(f"Berechne P(HATE) für {len(chunks)} Texte...")
    chunks['p_hate'] = score_with_pretrained(chunks['text'].tolist(), tokenizer, model, device)

    print(f"\nP(HATE) Verteilung:")
    print(f"  > 0.50:  {(chunks['p_hate'] > 0.50).sum()}")
    print(f"  > 0.30:  {(chunks['p_hate'] > 0.30).sum()}")
    print(f"  > 0.15:  {(chunks['p_hate'] > 0.15).sum()}")
    print(f"  ≤ 0.15:  {(chunks['p_hate'] <= 0.15).sum()}")

    # 5. Stratifizierte Stichprobe ziehen
    print(f"\nZiehe stratifizierte Stichprobe ({N_HIGH_HATE} hoch + {N_RANDOM} zufällig)...")
    high_hate = chunks[chunks['p_hate'] > HATE_THRESH]
    low_hate  = chunks[chunks['p_hate'] <= HATE_THRESH]

    n_high = min(N_HIGH_HATE, len(high_hate))
    sample_high = high_hate.nlargest(n_high, 'p_hate')

    # Zufällige Texte aus dem Rest (nicht aus high-hate)
    already_selected = set(sample_high.index)
    pool_random = chunks[~chunks.index.isin(already_selected)]
    n_rand = min(N_RANDOM, len(pool_random))
    sample_rand = pool_random.sample(n=n_rand, random_state=RANDOM_STATE)

    sample = pd.concat([sample_high, sample_rand]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    sample['id']    = sample.index + 1
    sample['label'] = ''   # Vom Annotator auszufüllen

    print(f"  Stichprobengröße: {len(sample)}")
    print(f"    davon P(HATE) > {HATE_THRESH}: {n_high}")
    print(f"    davon zufällig:                 {n_rand}")

    # 6. Speichern
    out_cols = ['id', 'doc_id', 'text', 'p_hate', 'label']
    sample[out_cols].to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(f"\n[OK] Annotationsstichprobe gespeichert: {OUT_CSV}")
    print(f"     {len(sample)} Texte — jetzt mit kodierung_tool.py annotieren.")
    print(f"\nNächster Schritt:")
    print(f"  python Data/annotation/kodierung_tool.py")


if __name__ == '__main__':
    main()
