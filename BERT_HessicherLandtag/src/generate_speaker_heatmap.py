"""
Heatmap: Top-Redner × Partei in HATE+Migrations-Dokumenten
Zeigt Top-20 Redner (gesamt) mit Parteizugehörigkeit als Farbkodierung.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
VIZ_DIR  = BASE_DIR / "Data" / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)
DPI = 300

PARTY_ORDER = ['AfD', 'CDU', 'SPD', 'GRÜNE', 'FDP', 'LINKE']
PARTY_COLORS = {
    'CDU':   '#555555',
    'SPD':   '#E3000F',
    'GRÜNE': '#46922a',
    'FDP':   '#D4B800',
    'LINKE': '#BE3075',
    'AfD':   '#009EE0',
}

df = pd.read_csv(BASE_DIR / "Data" / "evaluation" / "hate_migration_speakers.csv")
df = df[df['party'].isin(PARTY_ORDER)].sort_values('count', ascending=False)

# Top-20 Redner gesamt
top20 = df.head(20).copy()

# Pivot: Zeilen = Redner, Spalten = Parteien — Wert nur in der eigenen Partei-Spalte
pivot = pd.DataFrame(0, index=top20['name'], columns=PARTY_ORDER)
for _, row in top20.iterrows():
    if row['party'] in PARTY_ORDER:
        pivot.loc[row['name'], row['party']] = row['count']

# Heatmap
fig, ax = plt.subplots(figsize=(11, 9))

# Maske: alle Nullen ausblenden (transparent)
mask = pivot == 0

sns.heatmap(
    pivot,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    mask=mask,
    linewidths=0.5,
    linecolor='#eeeeee',
    cbar_kws={'label': 'Anzahl Vorkommen in HATE+Migrations-Dokumenten'},
    ax=ax,
    vmin=1,
)

# Partei-Spaltenköpfe einfärben
for tick in ax.get_xticklabels():
    party = tick.get_text()
    tick.set_color(PARTY_COLORS.get(party, 'black'))
    tick.set_fontweight('bold')
    tick.set_fontsize(11)

# Zeilenbeschriftung: Parteifarbe
for tick, name in zip(ax.get_yticklabels(), pivot.index):
    row = top20[top20['name'] == name]
    if not row.empty:
        party = row.iloc[0]['party']
        tick.set_color(PARTY_COLORS.get(party, 'black'))
    tick.set_fontsize(9)

ax.set_xlabel('Partei', fontsize=12, labelpad=10)
ax.set_ylabel('')
ax.set_title(
    'Top-20 Redner in HATE-Dokumenten mit Migrationsbezug\n(Farbe der Beschriftung = Parteizugehörigkeit)',
    fontweight='bold', fontsize=13, pad=15
)

# Legende
legend_patches = [
    mpatches.Patch(color=c, label=p)
    for p, c in PARTY_COLORS.items()
    if p in top20['party'].values
]
ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
          title='Partei', title_fontsize=9, framealpha=0.9)

plt.tight_layout()
out = VIZ_DIR / 'hate_migration_speaker_party_heatmap.png'
plt.savefig(out, dpi=DPI, bbox_inches='tight')
plt.close()

print(f"[OK] Gespeichert: {out}")
print(f"\nTop-20 Redner:")
print(top20[['name', 'party', 'count']].to_string(index=False))
