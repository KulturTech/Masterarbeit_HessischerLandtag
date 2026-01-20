"""
Word Cloud NUR der tatsaechlich negativ konnotierten Begriffe im Migrationskontext
"""
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Tatsaechlich negative/kritische Adjektive im Migrationskontext
TRULY_NEGATIVE_ADJECTIVES = {
    # Direkt negative Begriffe
    'illegal', 'illegale', 'illegalen', 'illegaler',
    'kriminell', 'kriminelle', 'kriminellen', 'krimineller',
    'gefährlich', 'gefährliche', 'gefährlichen', 'gefährlicher',
    'unkontrolliert', 'unkontrollierte', 'unkontrollierten', 'unkontrollierter',
    'massenhaft', 'massenhafte', 'massenhaften',
    'abgelehnt', 'abgelehnten', 'abgelehnte', 'abgelehnter',
    'ausreisepflichtig', 'ausreisepflichtige', 'ausreisepflichtigen',
    'gescheitert', 'gescheiterte', 'gescheiterten', 'gescheiterter',
    'falsch', 'falsche', 'falschen', 'falscher',
    'problematisch', 'problematische', 'problematischen',
    'straffällig', 'straffällige', 'straffälligen',
    'unerlaubt', 'unerlaubte', 'unerlaubten',
    'unberechtigt', 'unberechtigte', 'unberechtigten',
    'missbräuchlich', 'missbräuchliche', 'missbräuchlichen',
    'chaotisch', 'chaotische', 'chaotischen',
    'unklar', 'unklare', 'unklaren',
    'unsicher', 'unsichere', 'unsicheren',
    'ungeregelt', 'ungeregelte', 'ungeregelten',
    'irregulär', 'irreguläre', 'irregulären',
    'fremd', 'fremde', 'fremden', 'fremder',
    'unqualifiziert', 'unqualifizierte', 'unqualifizierten',
    'arbeitslos', 'arbeitslose', 'arbeitslosen',
    'unintegriert', 'unintegrierte', 'unintegrierten',
    'gewalttätig', 'gewalttätige', 'gewalttätigen',
    'radikal', 'radikale', 'radikalen', 'radikaler',
    'extremistisch', 'extremistische', 'extremistischen',
    'islamistisch', 'islamistische', 'islamistischen',
    'unverantwortlich', 'unverantwortliche',
    'schlecht', 'schlechte', 'schlechten', 'schlechter',
    'negativ', 'negative', 'negativen',
    'bedrohlich', 'bedrohliche', 'bedrohlichen',
    'überlastet', 'überlastete', 'überlasteten',
    'überfüllt', 'überfüllte', 'überfüllten',
    'massiv', 'massive', 'massiven',
}

# Lade die negativen Adjektive
df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\negative_adjectives_migration.csv')

print("="*60)
print("WORD CLOUD: NEGATIV KONNOTIERTE BEGRIFFE")
print("="*60)

# Filtere nur tatsaechlich negative Adjektive
df['adj_lower'] = df['adjective'].str.lower()
df_negative = df[df['adj_lower'].isin(TRULY_NEGATIVE_ADJECTIVES)]

print(f"Gesamt Adjektive: {len(df)}")
print(f"Davon negativ konnotiert: {len(df_negative)}")

# Zaehle
adj_counts = Counter(df_negative['adj_lower'])

print(f"\nTop 30 negativ konnotierte Adjektive:")
print("-"*40)
for adj, count in adj_counts.most_common(30):
    print(f"  {adj:25}: {count:4}x")

# Word Cloud erstellen
if len(adj_counts) > 0:
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='Reds',
        max_words=80,
        min_font_size=12,
        max_font_size=200,
        relative_scaling=0.4,
        prefer_horizontal=0.8
    ).generate_from_frequencies(dict(adj_counts))

    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negativ konnotierte Adjektive im Migrationskontext\n(Hessischer Landtag 2019-2023)',
              fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\wordcloud_truly_negative.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n[OK] Word Cloud gespeichert: {output_path}")

# Word Cloud nach Partei
print("\nErstelle Word Clouds pro Partei...")

# Lade Speaker-Daten
speaker_df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\speaker_adjectives_migration.csv')

# Extrahiere Adjektive
party_adj_list = []
for _, row in speaker_df.iterrows():
    party = row['party']
    adj_str = row['adjectives_str']
    if pd.notna(adj_str) and adj_str:
        for adj in str(adj_str).split(', '):
            adj = adj.strip().lower()
            if adj and adj in TRULY_NEGATIVE_ADJECTIVES:
                party_adj_list.append({'party': party, 'adjective': adj})

party_adj_df = pd.DataFrame(party_adj_list)

print(f"\nNegative Adjektive pro Partei (NER-basiert):")
print("-"*40)
for party in ['AfD', 'CDU', 'SPD', 'GRUENE', 'LINKE', 'FDP']:
    count = len(party_adj_df[party_adj_df['party'] == party])
    print(f"  {party:8}: {count:3}x")

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
        party_counts = Counter(party_data['adjective'])

        if len(party_counts) > 0:
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap=colors.get(party, 'viridis'),
                max_words=40,
                min_font_size=10,
                max_font_size=100
            ).generate_from_frequencies(dict(party_counts))

            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f'{party}\n({len(party_data)} negative Adjektive)',
                            fontsize=14, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, 'Keine Daten', ha='center', va='center', fontsize=12)
            axes[i].set_title(f'{party}', fontsize=14)
    else:
        axes[i].text(0.5, 0.5, 'Keine Daten', ha='center', va='center', fontsize=12)
        axes[i].set_title(f'{party}', fontsize=14)

    axes[i].axis('off')

plt.suptitle('Negativ konnotierte Adjektive im Migrationskontext nach Partei\n(Hessischer Landtag 2019-2023)',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

output_path2 = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\wordcloud_truly_negative_by_party.png'
plt.savefig(output_path2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"[OK] Word Cloud nach Partei gespeichert: {output_path2}")

# Speichere die gefilterten Daten
df_negative.to_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\truly_negative_adjectives.csv', index=False)
print(f"[OK] Gefilterte Daten gespeichert: truly_negative_adjectives.csv")

print("\n[OK] Fertig!")
