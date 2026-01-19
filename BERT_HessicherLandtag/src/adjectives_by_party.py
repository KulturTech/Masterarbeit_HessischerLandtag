"""
Analyse negativer Adjektive im Migrations-Kontext nach Parteien
===============================================================
Verknuepft negative Adjektive mit Partei-Erwaehnungen im selben Dokument.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

# Konfiguration
INPUT_PATH = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_migration_context.parquet'
OUTPUT_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1'
VIZ_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\visualizations'

# Parteien mit Farben
PARTIES = {
    'CDU': '#000000',      # Schwarz
    'SPD': '#E3000F',      # Rot
    'GRUENE': '#64A12D',   # Gruen
    'FDP': '#FFED00',      # Gelb
    'LINKE': '#BE3075',    # Magenta
    'AfD': '#009EE0'       # Blau
}

# Regex-Patterns fuer Parteien
PARTY_PATTERNS = {
    'CDU': r'\bCDU\b',
    'SPD': r'\bSPD\b',
    'GRUENE': r'\bGR[UÜ]NE[N]?\b|\bB[uü]ndnis\s*90\b',
    'FDP': r'\bFDP\b',
    'LINKE': r'\bLINKE[N]?\b|\bDie\s+Linke\b',
    'AfD': r'\bAfD\b'
}

print("="*80)
print("NEGATIVE ADJEKTIVE NACH PARTEIEN")
print("="*80)

# Lade Daten
print(f"\nLade Daten: {INPUT_PATH}")
try:
    df = pd.read_parquet(INPUT_PATH)
except FileNotFoundError:
    # Fallback auf CSV
    INPUT_PATH = INPUT_PATH.replace('.parquet', '.csv')
    df = pd.read_csv(INPUT_PATH)

print(f"[OK] {len(df):,} Adjektiv-Vorkommen geladen")

# Filtere nur negative Kontexte
df_negative = df[df['context_sentiment'] == 'negative'].copy()
print(f"[OK] {len(df_negative):,} Adjektive in negativem Kontext")

# Extrahiere Partei-Erwaehnungen aus dem Kontext
print("\nExtrahiere Partei-Erwaehnungen...")

for party, pattern in PARTY_PATTERNS.items():
    df_negative[party] = df_negative['context'].str.contains(pattern, case=False, regex=True, na=False)

# Zaehle Adjektive pro Partei
print("\nZaehle Adjektive pro Partei...")

party_adjectives = defaultdict(Counter)
party_total = Counter()

for _, row in df_negative.iterrows():
    lemma = row['lemma']
    for party in PARTIES.keys():
        if row[party]:
            party_adjectives[party][lemma] += 1
            party_total[party] += 1

# Ergebnisse ausgeben
print("\n" + "="*80)
print("ERGEBNISSE: TOP NEGATIVE ADJEKTIVE PRO PARTEI")
print("="*80)

for party in PARTIES.keys():
    count = party_total[party]
    if count > 0:
        print(f"\n{party} ({count} Adjektive in negativem Kontext):")
        print("-" * 40)
        for adj, c in party_adjectives[party].most_common(15):
            print(f"  {adj}: {c}")

# Speichere detaillierte Ergebnisse
print("\n" + "="*80)
print("SPEICHERE ERGEBNISSE")
print("="*80)

# CSV mit allen Daten
df_negative.to_csv(f'{OUTPUT_DIR}/negative_adjectives_with_parties.csv', index=False, encoding='utf-8')
print(f"[OK] negative_adjectives_with_parties.csv gespeichert")

# JSON Zusammenfassung
summary = {
    'total_negative_adjectives': len(df_negative),
    'by_party': {}
}

for party in PARTIES.keys():
    summary['by_party'][party] = {
        'total': party_total[party],
        'top_adjectives': dict(party_adjectives[party].most_common(20))
    }

with open(f'{OUTPUT_DIR}/adjectives_by_party_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[OK] adjectives_by_party_summary.json gespeichert")

# Visualisierungen
print("\n" + "="*80)
print("ERSTELLE VISUALISIERUNGEN")
print("="*80)

Path(VIZ_DIR).mkdir(parents=True, exist_ok=True)

# Plot 1: Anzahl negativer Adjektive pro Partei
fig, ax = plt.subplots(figsize=(10, 6))
parties_with_data = [p for p in PARTIES.keys() if party_total[p] > 0]
counts = [party_total[p] for p in parties_with_data]
colors = [PARTIES[p] for p in parties_with_data]

bars = ax.bar(parties_with_data, counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Partei')
ax.set_ylabel('Anzahl negativer Adjektive')
ax.set_title('Negative Adjektive im Migrations-Kontext nach Partei', fontsize=14, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/negative_adjectives_by_party_count.png', dpi=300, bbox_inches='tight')
print(f"[OK] negative_adjectives_by_party_count.png gespeichert")
plt.close()

# Plot 2: Heatmap der Top-Adjektive pro Partei
print("\nErstelle Heatmap...")

# Sammle alle Top-Adjektive
all_top_adjectives = set()
for party in parties_with_data:
    for adj, _ in party_adjectives[party].most_common(10):
        all_top_adjectives.add(adj)

all_top_adjectives = sorted(list(all_top_adjectives))

# Erstelle Matrix
heatmap_data = []
for adj in all_top_adjectives:
    row = [party_adjectives[party].get(adj, 0) for party in parties_with_data]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=all_top_adjectives, columns=parties_with_data)

# Plotte Heatmap
fig, ax = plt.subplots(figsize=(12, max(8, len(all_top_adjectives) * 0.4)))
sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='Reds', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Haeufigkeit'})
ax.set_xlabel('Partei')
ax.set_ylabel('Adjektiv')
ax.set_title('Top negative Adjektive im Migrations-Kontext nach Partei', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/negative_adjectives_party_heatmap.png', dpi=300, bbox_inches='tight')
print(f"[OK] negative_adjectives_party_heatmap.png gespeichert")
plt.close()

# Plot 3: Top 5 Adjektive pro Partei (Grouped Bar Chart)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, party in enumerate(PARTIES.keys()):
    ax = axes[idx]
    top_5 = party_adjectives[party].most_common(5)

    if top_5:
        adjectives = [a[0] for a in top_5]
        counts = [a[1] for a in top_5]

        bars = ax.barh(adjectives[::-1], counts[::-1], color=PARTIES[party], edgecolor='black', alpha=0.8)
        ax.set_xlabel('Haeufigkeit')
        ax.set_title(f'{party}', fontsize=12, fontweight='bold')
        ax.bar_label(bars, padding=3)
    else:
        ax.text(0.5, 0.5, 'Keine Daten', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{party}', fontsize=12, fontweight='bold')

plt.suptitle('Top 5 negative Adjektive im Migrations-Kontext pro Partei', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/top5_negative_adjectives_per_party.png', dpi=300, bbox_inches='tight')
print(f"[OK] top5_negative_adjectives_per_party.png gespeichert")
plt.close()

print("\n" + "="*80)
print("ANALYSE ABGESCHLOSSEN")
print("="*80)
print(f"\nErgebnisse in: {OUTPUT_DIR}")
print(f"Visualisierungen in: {VIZ_DIR}")
