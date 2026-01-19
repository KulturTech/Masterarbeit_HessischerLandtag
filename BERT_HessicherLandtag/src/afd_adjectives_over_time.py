"""
Zeitliche Entwicklung negativer Adjektive bei der AfD im Migrations-Kontext
===========================================================================
Analysiert wie sich die Verwendung negativer Adjektive ueber die Jahre entwickelt.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
from pathlib import Path

# Konfiguration
OUTPUT_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1'
VIZ_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\visualizations'

# Monatsnamen fuer Konvertierung
MONTH_MAP = {
    'januar': 1, 'februar': 2, 'märz': 3, 'maerz': 3, 'april': 4,
    'mai': 5, 'juni': 6, 'juli': 7, 'august': 8,
    'september': 9, 'oktober': 10, 'november': 11, 'dezember': 12
}

def extract_date_from_text(text):
    """Extrahiert Datum aus dem Text (typischerweise am Anfang der Protokolle)"""
    if not isinstance(text, str):
        return None

    # Suche nach deutschem Datumsformat: "18. Januar 2019"
    pattern = r'(\d{1,2})\.\s*(Januar|Februar|M[aä]rz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s*(20\d{2})'
    match = re.search(pattern, text[:1000], re.IGNORECASE)

    if match:
        day = int(match.group(1))
        month_name = match.group(2).lower()
        year = int(match.group(3))
        month = MONTH_MAP.get(month_name, 1)
        return pd.Timestamp(year=year, month=month, day=day)

    return None

print("="*80)
print("ZEITLICHE ENTWICKLUNG: AfD NEGATIVE ADJEKTIVE IM MIGRATIONS-KONTEXT")
print("="*80)

# Lade Adjektiv-Daten mit Parteien
print("\nLade Daten...")
df_adj = pd.read_csv(f'{OUTPUT_DIR}/negative_adjectives_with_parties.csv')
print(f"[OK] {len(df_adj):,} negative Adjektive geladen")

# Filtere nur AfD
df_afd = df_adj[df_adj['AfD'] == True].copy()
print(f"[OK] {len(df_afd):,} Adjektive mit AfD-Erwaehnung")

# Lade Originaldaten fuer Datums-Extraktion
print("\nLade Originaldaten fuer Datums-Extraktion...")
df_orig = pd.read_parquet(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_clean.parquet')

# Erstelle Mapping: doc_id -> Datum
print("Extrahiere Daten aus Dokumenten...")
doc_dates = {}
for _, row in df_orig.iterrows():
    doc_id = row['doc_id']
    if doc_id not in doc_dates:
        date = extract_date_from_text(row['text'])
        if date:
            doc_dates[doc_id] = date

print(f"[OK] {len(doc_dates):,} Dokumente mit Datum gefunden")

# Fuege Datum zu AfD-Daten hinzu
df_afd['date'] = df_afd['doc_id'].map(doc_dates)
df_afd = df_afd.dropna(subset=['date'])
df_afd['year'] = df_afd['date'].dt.year
df_afd['year_month'] = df_afd['date'].dt.to_period('M')

print(f"[OK] {len(df_afd):,} Adjektive mit Datum")

# Analysiere nach Jahren
print("\n" + "="*80)
print("ERGEBNISSE: ENTWICKLUNG UEBER DIE JAHRE")
print("="*80)

years = sorted(df_afd['year'].unique())
year_adjectives = defaultdict(Counter)
year_totals = Counter()

for _, row in df_afd.iterrows():
    year = row['year']
    lemma = row['lemma']
    year_adjectives[year][lemma] += 1
    year_totals[year] += 1

for year in years:
    total = year_totals[year]
    print(f"\n{int(year)} ({total} Adjektive):")
    print("-" * 40)
    for adj, count in year_adjectives[year].most_common(10):
        print(f"  {adj}: {count}")

# Speichere Ergebnisse
print("\n" + "="*80)
print("SPEICHERE ERGEBNISSE")
print("="*80)

df_afd.to_csv(f'{OUTPUT_DIR}/afd_negative_adjectives_with_dates.csv', index=False, encoding='utf-8')
print(f"[OK] afd_negative_adjectives_with_dates.csv gespeichert")

# Visualisierungen
print("\n" + "="*80)
print("ERSTELLE VISUALISIERUNGEN")
print("="*80)

Path(VIZ_DIR).mkdir(parents=True, exist_ok=True)

# Plot 1: Gesamtzahl negativer Adjektive pro Jahr
fig, ax = plt.subplots(figsize=(10, 6))
years_list = [int(y) for y in years]
totals_list = [year_totals[y] for y in years]

bars = ax.bar(years_list, totals_list, color='#009EE0', edgecolor='black', alpha=0.8)
ax.set_xlabel('Jahr')
ax.set_ylabel('Anzahl negativer Adjektive')
ax.set_title('AfD: Negative Adjektive im Migrations-Kontext pro Jahr', fontsize=14, fontweight='bold')
ax.set_xticks(years_list)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/afd_negative_adjectives_per_year.png', dpi=300, bbox_inches='tight')
print(f"[OK] afd_negative_adjectives_per_year.png gespeichert")
plt.close()

# Plot 2: Entwicklung spezifischer Adjektive ueber Zeit
# Fokus auf besonders relevante Adjektive
key_adjectives = ['illegal', 'kriminell', 'gefaehrlich', 'ausreisepflichtig', 'falsch', 'fremd']

# Zaehle pro Jahr
adj_per_year = defaultdict(lambda: defaultdict(int))
for _, row in df_afd.iterrows():
    year = int(row['year'])
    lemma = row['lemma']
    if lemma in key_adjectives:
        adj_per_year[lemma][year] += 1

fig, ax = plt.subplots(figsize=(12, 7))

for adj in key_adjectives:
    if any(adj_per_year[adj].values()):
        y_values = [adj_per_year[adj].get(y, 0) for y in years_list]
        ax.plot(years_list, y_values, marker='o', linewidth=2, markersize=8, label=adj)

ax.set_xlabel('Jahr')
ax.set_ylabel('Haeufigkeit')
ax.set_title('AfD: Entwicklung spezifischer negativer Adjektive im Migrations-Kontext', fontsize=14, fontweight='bold')
ax.set_xticks(years_list)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/afd_key_adjectives_over_time.png', dpi=300, bbox_inches='tight')
print(f"[OK] afd_key_adjectives_over_time.png gespeichert")
plt.close()

# Plot 3: Heatmap Adjektive pro Jahr
top_adjectives = []
for adj, count in Counter(df_afd['lemma']).most_common(15):
    top_adjectives.append(adj)

heatmap_data = []
for adj in top_adjectives:
    row = [year_adjectives[y].get(adj, 0) for y in years]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=top_adjectives, columns=[int(y) for y in years])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='Blues', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Haeufigkeit'})
ax.set_xlabel('Jahr')
ax.set_ylabel('Adjektiv')
ax.set_title('AfD: Top 15 negative Adjektive im Migrations-Kontext pro Jahr', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/afd_adjectives_heatmap_years.png', dpi=300, bbox_inches='tight')
print(f"[OK] afd_adjectives_heatmap_years.png gespeichert")
plt.close()

# Plot 4: Anteil bestimmter Adjektive (illegal, kriminell) am Gesamt
fig, ax = plt.subplots(figsize=(10, 6))

negative_terms = ['illegal', 'kriminell', 'gefaehrlich', 'ausreisepflichtig']
percentages = []

for year in years_list:
    total = year_totals[year]
    neg_count = sum(year_adjectives[year].get(adj, 0) for adj in negative_terms)
    pct = (neg_count / total * 100) if total > 0 else 0
    percentages.append(pct)

bars = ax.bar(years_list, percentages, color='#dc3545', edgecolor='black', alpha=0.8)
ax.set_xlabel('Jahr')
ax.set_ylabel('Anteil (%)')
ax.set_title('AfD: Anteil stark negativer Adjektive\n(illegal, kriminell, gefaehrlich, ausreisepflichtig)',
             fontsize=14, fontweight='bold')
ax.set_xticks(years_list)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/afd_strong_negative_percentage.png', dpi=300, bbox_inches='tight')
print(f"[OK] afd_strong_negative_percentage.png gespeichert")
plt.close()

print("\n" + "="*80)
print("ANALYSE ABGESCHLOSSEN")
print("="*80)
print(f"\nErgebnisse in: {OUTPUT_DIR}")
print(f"Visualisierungen in: {VIZ_DIR}")
