"""
Zusammenführen aller verfügbaren Trainingsdaten
================================================
Kombiniert labeled_data_combined.csv (externe + Hessen-Daten)
mit annotation_sample.csv (manuell annotiert) zu einem
einzigen, deduplizierten Trainingsdatensatz.
"""

import pandas as pd
from pathlib import Path

BASE_DIR   = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
OUTPUT_CSV = BASE_DIR / "Data" / "training" / "labeled_data_all.csv"

SOURCES = [
    BASE_DIR / "Data" / "training"    / "labeled_data_combined.csv",
    BASE_DIR / "Data" / "annotation"  / "annotation_sample.csv",
]

print("=" * 60)
print("TRAININGSDATEN ZUSAMMENFÜHREN")
print("=" * 60)

frames = []
for path in SOURCES:
    df = pd.read_csv(path)
    df = df[df['label'].isin(['HATE', 'NON_HATE'])][['text', 'label']].copy()
    counts = df['label'].value_counts().to_dict()
    print(f"\n{path.name}:")
    print(f"  HATE={counts.get('HATE',0)}, NON_HATE={counts.get('NON_HATE',0)}, gesamt={len(df)}")
    frames.append(df)

combined = pd.concat(frames, ignore_index=True)
print(f"\nVor Deduplizierung: {len(combined)} Zeilen")

# Duplikate anhand des Textes entfernen (ersten Eintrag behalten)
combined = combined.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
print(f"Nach Deduplizierung: {len(combined)} Zeilen")

counts = combined['label'].value_counts()
print(f"\nEndgültige Verteilung:")
print(f"  HATE    : {counts.get('HATE', 0)}")
print(f"  NON_HATE: {counts.get('NON_HATE', 0)}")
print(f"  Gesamt  : {len(combined)}")
print(f"  HATE-Anteil: {counts.get('HATE',0)/len(combined)*100:.1f}%")

combined.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n[OK] Gespeichert: {OUTPUT_CSV}")
print("\nNächster Schritt: retrain.py mit dem neuen Datensatz ausführen.")
print(f"  Dazu ANNOTATION_CSV in retrain.py auf:\n  {OUTPUT_CSV}")
