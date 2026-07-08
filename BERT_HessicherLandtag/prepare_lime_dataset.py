import sys
import pandas as pd
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")

df = pd.read_csv(BASE_DIR / 'Data/evaluation/lexikon_treffer_alle.csv')

# critical=True ausschließen (zitierende Verwendung, nicht Eigenverwendung)
df_eigen = df[df['critical'] == False].copy()
print(f"Gesamt Lexikon-Treffer: {len(df)}")
print(f"Eigenverwendung (nicht kritisch): {len(df_eigen)}")
print(f"Label-Verteilung:\n{df_eigen['label'].value_counts()}")

# Spalte 'context' -> 'text' umbenennen fuer LIME-Kompatibilitaet
df_eigen = df_eigen.rename(columns={'context': 'text'})

# Top-N nach score (NON_HATE score = Konfidenz, also absteigend fuer sicherste Klassifikationen)
TOP_N = 10
top = df_eigen.nlargest(TOP_N, 'score')

print(f"\nTop-{TOP_N} Texte (nach BERT-Score):")
for i, (_, row) in enumerate(top.iterrows(), 1):
    print(f"{i}. [{row['label']} {row['score']:.4f}] {row.get('speaker','?')} ({row.get('party','?')})")
    print(f"   {str(row['text'])[:100]}...")

# Speichern
out = BASE_DIR / 'Data/evaluation/hate_speeches_for_lime.csv'
top.to_csv(out, index=False, encoding='utf-8-sig')
print(f"\nGespeichert: {out}")
