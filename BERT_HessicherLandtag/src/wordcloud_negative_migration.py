"""
Word Cloud der Begriffe im negativen Kontext fuer Migration
"""
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Lade die negativen Adjektive
df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\negative_adjectives_migration.csv')

print("="*60)
print("WORD CLOUD: NEGATIVE BEGRIFFE IM MIGRATIONSKONTEXT")
print("="*60)
print(f"Geladene Datensaetze: {len(df)}")

# Zaehle Adjektive
adj_counts = Counter(df['adjective'].str.lower())

print(f"\nTop 30 haeufigste negative Adjektive:")
print("-"*40)
for adj, count in adj_counts.most_common(30):
    print(f"  {adj:20}: {count:4}x")

# Erstelle Word Cloud
print("\nErstelle Word Cloud...")

# Frequenz-Dictionary fuer WordCloud
freq_dict = dict(adj_counts)

# Word Cloud erstellen
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    colormap='Reds',  # Rot-Toene fuer negative Begriffe
    max_words=100,
    min_font_size=10,
    max_font_size=150,
    relative_scaling=0.5,
    prefer_horizontal=0.7
).generate_from_frequencies(freq_dict)

# Speichern
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Adjektive im Migrationskontext (Hessischer Landtag 2019-2023)',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

output_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\wordcloud_negative_migration.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"[OK] Word Cloud gespeichert: {output_path}")

# Zusaetzlich: Word Cloud nach Partei
print("\nErstelle Word Clouds pro Partei...")

# Lade Speaker-Daten und extrahiere Adjektive
speaker_df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\speaker_adjectives_migration.csv')

# Erstelle DataFrame mit einzelnen Adjektiven pro Partei
party_adj_list = []
for _, row in speaker_df.iterrows():
    party = row['party']
    adj_str = row['adjectives_str']
    if pd.notna(adj_str) and adj_str:
        for adj in str(adj_str).split(', '):
            adj = adj.strip()
            if adj:
                party_adj_list.append({'party': party, 'adjective': adj})

party_adj_df = pd.DataFrame(party_adj_list)

parties = ['AfD', 'CDU', 'SPD', 'GRUENE', 'LINKE', 'FDP']
colors = {
    'AfD': 'Blues',
    'CDU': 'Greys',
    'SPD': 'Reds',
    'GRUENE': 'Greens',
    'LINKE': 'Purples',
    'FDP': 'YlOrBr'
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, party in enumerate(parties):
    party_data = party_adj_df[party_adj_df['party'] == party]

    if len(party_data) > 0:
        party_counts = Counter(party_data['adjective'].str.lower())

        if len(party_counts) > 0:
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=colors.get(party, 'viridis'),
                max_words=50,
                min_font_size=8,
                max_font_size=80
            ).generate_from_frequencies(dict(party_counts))

            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f'{party}\n({len(party_data)} Adjektive)',
                            fontsize=14, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'Keine Daten', ha='center', va='center')
            axes[i].set_title(f'{party}', fontsize=14)
    else:
        axes[i].text(0.5, 0.5, 'Keine Daten', ha='center', va='center')
        axes[i].set_title(f'{party}', fontsize=14)

    axes[i].axis('off')

plt.suptitle('Negative Adjektive im Migrationskontext nach Partei\n(Hessischer Landtag 2019-2023)',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

output_path2 = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\wordcloud_negative_by_party.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"[OK] Word Cloud nach Partei gespeichert: {output_path2}")

print("\n[OK] Fertig!")
