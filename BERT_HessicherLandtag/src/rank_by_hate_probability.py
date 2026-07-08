"""
Ranking aller Dokumente nach HATE-Wahrscheinlichkeit
=====================================================
Da der Klassifikator (Schwellenwert 0.5) keine HATE-Dokumente findet,
werden hier alle Dokumente nach ihrer HATE-Wahrscheinlichkeit (1 - NON_HATE-Score)
absteigend sortiert. Die Top-N werden zur manuellen Prüfung ausgegeben.
"""

import pandas as pd
from pathlib import Path

INPUT_PATH  = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_classified.parquet")
OUTPUT_DIR  = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\evaluation")
TOP_N       = 200   # Wie viele der verdächtigsten Dokumente ausgeben
THRESHOLD   = 0.35  # Alles mit HATE-Wahrscheinlichkeit > 35% als verdächtig markieren

df = pd.read_parquet(INPUT_PATH)

# Abstimmungslisten herausfiltern (enthalten strukturierte Namensspalten)
ABSTIMMUNG_MARKER = [
    "Name der/des Abgeordneten",
    "ja  nein  enthalten  gefehlt",
    "ja nein enthalten gefehlt",
]
is_abstimmung = df['text'].apply(
    lambda t: any(m in t for m in ABSTIMMUNG_MARKER)
)
n_filtered = is_abstimmung.sum()
df = df[~is_abstimmung].copy()

# HATE-Wahrscheinlichkeit berechnen (Modell gibt NON_HATE-Score aus)
df['hate_probability'] = 1 - df['score']

# Absteigend sortieren
df_ranked = df.sort_values('hate_probability', ascending=False).reset_index(drop=True)
df_ranked['rank'] = df_ranked.index + 1

print("=" * 70)
print("RANKING NACH HATE-WAHRSCHEINLICHKEIT (ohne Abstimmungslisten)")
print("=" * 70)
print(f"Gesamt: {len(df_ranked)} Dokumente ({n_filtered} Abstimmungslisten gefiltert)")
print(f"\nVerteilung der HATE-Wahrscheinlichkeit:")
for threshold in [0.40, 0.35, 0.30, 0.25, 0.20]:
    count = (df_ranked['hate_probability'] >= threshold).sum()
    print(f"  >= {threshold*100:.0f}%: {count} Dokumente")

print(f"\nTop {TOP_N} verdächtigste Dokumente:")
print("-" * 70)
top = df_ranked.head(TOP_N)[['rank', 'doc_id', 'hate_probability', 'text']]
for _, row in top.iterrows():
    preview = row['text'][:200].replace('\n', ' ').strip().encode('cp1252', errors='replace').decode('cp1252')
    print(f"\n[#{row['rank']}] HATE-W'keit: {row['hate_probability']:.3f}")
    print(f"  Doc: {row['doc_id'][:60]}...")
    print(f"  Text: {preview}...")

# Speichern
out_all = OUTPUT_DIR / "ranked_by_hate_probability.csv"
df_ranked[['rank', 'doc_id', 'hate_probability', 'text']].to_csv(out_all, index=False, encoding='utf-8-sig')
print(f"\n[OK] Alle {len(df_ranked)} Dokumente gespeichert: {out_all}")

# Nur verdächtige (über Schwellenwert)
df_suspicious = df_ranked[df_ranked['hate_probability'] >= THRESHOLD]
out_suspicious = OUTPUT_DIR / f"suspicious_hate_threshold_{int(THRESHOLD*100)}.csv"
df_suspicious[['rank', 'doc_id', 'hate_probability', 'text']].to_csv(out_suspicious, index=False, encoding='utf-8-sig')
print(f"[OK] {len(df_suspicious)} verdächtige Dok. (>={THRESHOLD*100:.0f}%) gespeichert: {out_suspicious}")
print(f"\nHinweis: {n_filtered} Abstimmungslisten wurden herausgefiltert.")
