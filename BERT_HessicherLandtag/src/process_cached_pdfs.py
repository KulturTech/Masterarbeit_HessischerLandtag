"""
Ältere Protokolle (WP 16-19) verarbeiten & Annotations-Sample erweitern
========================================================================
1. Extrahiert Text aus allen PDFs in Data/protocols_pdf_cache/
2. Filtert TOC/Kopfzeilen heraus
3. Extrahiert Absätze mit Migrations-Keywords (+ Kontext)
4. Bewertet mit dem vortrainierten dehatebert-mono-german
5. Fügt Top-Kandidaten zu Data/annotation/annotation_sample.csv hinzu
   (keine Duplikate, keine bereits annotierten Texte werden überschrieben)

Ausführung:
  python src/process_cached_pdfs.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Pfade & Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR        = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
CACHE_DIR       = BASE_DIR / "Data" / "protocols_pdf_cache"
ANNOTATION_CSV  = BASE_DIR / "Data" / "annotation" / "annotation_sample.csv"
SCORE_CACHE     = BASE_DIR / "Data" / "annotation" / "paragraph_scores_cache.parquet"

PRETRAINED_MODEL = "Hate-speech-CNERG/dehatebert-mono-german"
BATCH_SIZE       = 16
MAX_LENGTH       = 256
HATE_THRESH      = 0.15   # Schwelle für "hohes HATE-Potenzial"
N_NEW_HIGH       = 150    # Max. neue Hochrisiko-Texte hinzufügen
N_NEW_RANDOM     = 50     # Max. neue Zufallstexte hinzufügen
RANDOM_STATE     = 42

# ---------------------------------------------------------------------------
# Migrations-Keywords
# ---------------------------------------------------------------------------
_MIGRATION_RE = re.compile('|'.join([
    r'\b(immigrant\w*|einwanderer\w*|einwanderung\w*|zuwander\w*)\b',
    r'\b(migrant\w*|migration\w*)\b',
    r'\b(flüchtling\w*|asyl\w*)\b',
    r'\b(ausländer\w*|ausländisch\w*)\b',
    r'\b(geflüchtete\w*|schutzsuchende\w*)\b',
    r'\b(integration\w*)\b',
    r'\b(abschiebung\w*|rückführung\w*)\b',
    r'\b(grenzsicherung\w*|grenzschutz\w*)\b',
    r'\b(asylpolitik\w*|migrationspolitik\w*|ausländerpolitik\w*)\b',
]), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'(?m)^\s*\d{1,5}\s*$', ' ', text)   # Seitenzahlen
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def is_toc(text: str) -> bool:
    if not isinstance(text, str):
        return True
    t = text.strip()
    lines = [l for l in t.splitlines() if l.strip()]
    if not lines or len(t) < 100:
        return True
    dotted = sum(1 for l in lines if re.search(r'\.{4,}\s*\d*\s*$', l))
    return (dotted / len(lines)) > 0.3


def extract_migration_paragraphs(doc_id: str, text: str) -> list[dict]:
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


def score_texts(texts: list[str], tokenizer, model, device) -> np.ndarray:
    hate_idx = next(k for k, v in model.config.id2label.items() if v == 'HATE')
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
# Schritt 1: PDFs extrahieren
# ---------------------------------------------------------------------------
def extract_from_pdfs() -> pd.DataFrame:
    pdfs = sorted(CACHE_DIR.glob("*.pdf"))
    print(f"  {len(pdfs)} PDFs gefunden in {CACHE_DIR.name}/")

    rows = []
    errors = 0
    for i, pdf in enumerate(tqdm(pdfs, desc="  PDFs lesen"), 1):
        try:
            doc = pymupdf.open(str(pdf))
            full_text = "\n".join(page.get_text() for page in doc)
            doc.close()
            full_text = clean_text(full_text)
            if not is_toc(full_text):
                rows.extend(extract_migration_paragraphs(pdf.stem, full_text))
        except Exception as e:
            errors += 1

    df = pd.DataFrame(rows).drop_duplicates(subset='text').reset_index(drop=True)
    if errors:
        print(f"  {errors} PDFs konnten nicht gelesen werden")
    print(f"  {len(df)} Migrations-Absätze extrahiert")
    return df


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PROTOKOLLE WP 16-19 VERARBEITEN")
    print("=" * 70)

    # 1. PDFs extrahieren
    print("\nSchritt 1: Text aus PDFs extrahieren...")
    new_chunks = extract_from_pdfs()

    if len(new_chunks) == 0:
        print("Keine Migrations-Absätze gefunden.")
        return

    # 2. Bereits vorhandene Texte laden (aus WP 20 + bisherigem Sample)
    existing_texts = set()
    if ANNOTATION_CSV.exists():
        existing_sample = pd.read_csv(ANNOTATION_CSV)
        existing_texts.update(existing_sample['text'].str.strip().tolist())
        print(f"\nBestehendes Sample: {len(existing_sample)} Texte "
              f"({existing_sample['label'].isin(['HATE','NON_HATE']).sum()} bereits annotiert)")

    # Duplikate mit bestehendem Sample entfernen
    new_chunks = new_chunks[~new_chunks['text'].str.strip().isin(existing_texts)].copy()
    print(f"Neue (noch nicht im Sample): {len(new_chunks)} Absätze")

    if len(new_chunks) == 0:
        print("Keine neuen Texte hinzuzufügen.")
        return

    # 3. Score-Cache laden oder neu berechnen
    print("\nSchritt 2: P(HATE) berechnen...")
    new_chunks['p_hate'] = np.nan

    if SCORE_CACHE.exists():
        cached = pd.read_parquet(SCORE_CACHE)[['text', 'p_hate']]
        new_chunks = new_chunks.merge(cached, on='text', how='left',
                                      suffixes=('_old', ''))
        if 'p_hate_old' in new_chunks.columns:
            new_chunks['p_hate'] = new_chunks['p_hate'].fillna(new_chunks['p_hate_old'])
            new_chunks = new_chunks.drop(columns=['p_hate_old'])
        already_scored = new_chunks['p_hate'].notna().sum()
        print(f"  {already_scored} Scores aus Cache, "
              f"{new_chunks['p_hate'].isna().sum()} neu zu berechnen")

    to_score_mask = new_chunks['p_hate'].isna()
    if to_score_mask.sum() > 0:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Lade Modell ({device})...")
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            PRETRAINED_MODEL).to(device)
        model.eval()

        scores = score_texts(
            new_chunks[to_score_mask]['text'].tolist(), tokenizer, model, device
        )
        new_chunks.loc[to_score_mask, 'p_hate'] = scores

        # Cache aktualisieren
        updated_cache = pd.concat([
            pd.read_parquet(SCORE_CACHE) if SCORE_CACHE.exists() else pd.DataFrame(),
            new_chunks[['text', 'p_hate']]
        ]).drop_duplicates(subset='text')
        updated_cache.to_parquet(SCORE_CACHE)
        print(f"  Cache aktualisiert: {len(updated_cache)} Einträge")

    print(f"\n  P(HATE) Verteilung (neue Texte):")
    print(f"    > 0.50:  {(new_chunks['p_hate'] > 0.50).sum()}")
    print(f"    > 0.30:  {(new_chunks['p_hate'] > 0.30).sum()}")
    print(f"    > 0.15:  {(new_chunks['p_hate'] > 0.15).sum()}")
    print(f"    <= 0.15: {(new_chunks['p_hate'] <= 0.15).sum()}")

    # 4. Stratifizierte Auswahl neuer Texte
    print(f"\nSchritt 3: Neue Texte auswählen...")
    high = new_chunks[new_chunks['p_hate'] > HATE_THRESH]
    rest = new_chunks[new_chunks['p_hate'] <= HATE_THRESH]

    n_high = min(N_NEW_HIGH, len(high))
    n_rand = min(N_NEW_RANDOM, len(rest))

    sample_high = high.nlargest(n_high, 'p_hate')
    sample_rand = rest.sample(n=n_rand, random_state=RANDOM_STATE)

    new_sample = pd.concat([sample_high, sample_rand]) \
        .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    new_sample['label'] = ''

    print(f"  Neue Texte hinzufügen: {len(new_sample)} "
          f"({n_high} hoch-P(HATE), {n_rand} zufällig)")

    # 5. An bestehendes Sample anhängen
    if ANNOTATION_CSV.exists():
        existing_sample = pd.read_csv(ANNOTATION_CSV)
        # IDs fortsetzen
        max_id = existing_sample['id'].max() if 'id' in existing_sample.columns else 0
        new_sample['id'] = range(int(max_id) + 1, int(max_id) + 1 + len(new_sample))
        combined = pd.concat([existing_sample, new_sample[existing_sample.columns]],
                             ignore_index=True)
    else:
        new_sample['id'] = range(1, len(new_sample) + 1)
        combined = new_sample

    combined.to_csv(ANNOTATION_CSV, index=False, encoding='utf-8')

    total     = len(combined)
    annotated = combined['label'].isin(['HATE', 'NON_HATE']).sum()
    hate      = (combined['label'] == 'HATE').sum()
    print(f"\n[OK] Sample gespeichert: {ANNOTATION_CSV}")
    print(f"     Gesamt: {total} Texte")
    print(f"     Bereits annotiert: {annotated} ({hate} HATE, {annotated-hate} NON_HATE)")
    print(f"     Noch zu annotieren: {total - annotated}")
    print(f"\nNächster Schritt:")
    print(f"  python Data/annotation/kodierung_tool.py")


if __name__ == '__main__':
    main()
