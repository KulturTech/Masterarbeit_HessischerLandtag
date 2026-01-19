"""
Analyse negativer Adjektive nach Redner/Partei mittels NER-aehnlicher Extraktion
================================================================================
Identifiziert welche Partei welche Adjektive im Migrations-Kontext VERWENDET
(nicht nur erwaehnt), basierend auf Redner-Erkennung in Parlamentsprotokollen.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import re
import json
from pathlib import Path
from tqdm import tqdm

# Konfiguration
OUTPUT_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1'
VIZ_DIR = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\visualizations'

# Partei-Normalisierung
PARTY_NORMALIZE = {
    'CDU': 'CDU',
    'SPD': 'SPD',
    'BÜNDNIS 90/DIE GRÜNEN': 'GRUENE',
    'BÜNDNIS 90/ DIE GRÜNEN': 'GRUENE',
    'B90/GRÜNE': 'GRUENE',
    'DIE GRÜNEN': 'GRUENE',
    'GRÜNE': 'GRUENE',
    'DIE LINKE': 'LINKE',
    'LINKE': 'LINKE',
    'AfD': 'AfD',
    'AFD': 'AfD',
    'Freie Demokraten': 'FDP',
    'FDP': 'FDP',
}

# Partei-Farben
PARTY_COLORS = {
    'CDU': '#000000',
    'SPD': '#E3000F',
    'GRUENE': '#64A12D',
    'FDP': '#FFED00',
    'LINKE': '#BE3075',
    'AfD': '#009EE0'
}

# Regex fuer Redner-Erkennung: "Name (Partei):"
SPEAKER_PATTERN = re.compile(
    r'([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+)\s*\(([^)]+)\)\s*:',
    re.UNICODE
)

# Adjektive (aus dem Hauptskript)
GERMAN_ADJECTIVES = {
    'illegal', 'illegalen', 'illegaler', 'illegale',
    'kriminell', 'kriminelle', 'kriminellen', 'krimineller',
    'gefaehrlich', 'gefährlich', 'gefährliche', 'gefährlichen',
    'unkontrolliert', 'unkontrollierte', 'unkontrollierten',
    'massenhaft', 'massenhafte', 'massenhaften',
    'problematisch', 'problematische', 'problematischen',
    'fremd', 'fremde', 'fremden', 'fremder',
    'radikal', 'radikale', 'radikalen',
    'ausreisepflichtig', 'ausreisepflichtige', 'ausreisepflichtigen',
    'abgelehnt', 'abgelehnte', 'abgelehnten',
    'straffaellig', 'straffällig', 'straffällige',
    'legal', 'legale', 'legalen',
    'qualifiziert', 'qualifizierte', 'qualifizierten',
    'integriert', 'integrierte', 'integrierten',
    'humanitaer', 'humanitär', 'humanitäre', 'humanitären',
    'schutzbedürftig', 'schutzbedürftige',
    'gescheitert', 'gescheiterte', 'gescheiterten',
    'falsch', 'falsche', 'falschen',
    'sicher', 'sichere', 'sicheren',
    'gut', 'gute', 'guten',
    'wichtig', 'wichtige', 'wichtigen',
    'notwendig', 'notwendige', 'notwendigen',
}

# Migrations-Keywords
MIGRATION_KEYWORDS = re.compile(
    r'\b(migrant|migranten|migration|flüchtling|flüchtlinge|asyl|'
    r'einwander|zuwander|ausländer|geflüchtete|schutzsuchende)\w*\b',
    re.IGNORECASE
)

def normalize_party(party_str):
    """Normalisiert Parteinamen"""
    party_str = party_str.strip()
    for key, value in PARTY_NORMALIZE.items():
        if key.lower() in party_str.lower():
            return value
    return None

def extract_adjectives(text):
    """Extrahiert Adjektive aus Text"""
    words = text.lower().split()
    adjectives = []
    for word in words:
        clean = re.sub(r'[^\wäöüß-]', '', word)
        if clean in GERMAN_ADJECTIVES:
            adjectives.append(clean)
    return adjectives

def has_migration_context(text):
    """Prueft ob Text Migrations-Bezug hat"""
    return bool(MIGRATION_KEYWORDS.search(text))

print("="*80)
print("ADJEKTIVE NACH REDNER/PARTEI (NER-BASIERT)")
print("="*80)

# Lade Daten
print("\nLade Originaldaten...")
df = pd.read_parquet(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_clean.parquet')
print(f"[OK] {len(df):,} Dokumente geladen")

# Verarbeite alle Texte
print("\nExtrahiere Redner und deren Adjektive im Migrations-Kontext...")

# Speichere: Partei -> Adjektive
party_adjectives = defaultdict(Counter)
party_speakers = defaultdict(set)
speaker_adjectives = defaultdict(Counter)

# Sammle alle Redner-Segmente
all_segments = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verarbeite Dokumente"):
    text = row['text']

    # Finde alle Redner-Wechsel
    matches = list(SPEAKER_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        speaker_name = match.group(1).strip()
        party_raw = match.group(2).strip()
        party = normalize_party(party_raw)

        if party is None:
            continue

        # Extrahiere Text bis zum naechsten Redner
        start = match.end()
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        speech_text = text[start:end]

        # Pruefe auf Migrations-Kontext
        if has_migration_context(speech_text):
            adjectives = extract_adjectives(speech_text)

            for adj in adjectives:
                party_adjectives[party][adj] += 1
                speaker_adjectives[speaker_name][adj] += 1

            party_speakers[party].add(speaker_name)

            if adjectives:
                all_segments.append({
                    'speaker': speaker_name,
                    'party': party,
                    'adjectives': adjectives,
                    'text_preview': speech_text[:200]
                })

# Ergebnisse
print("\n" + "="*80)
print("ERGEBNISSE: ADJEKTIVE NACH PARTEI (WER SPRICHT)")
print("="*80)

party_totals = {p: sum(c.values()) for p, c in party_adjectives.items()}

for party in sorted(PARTY_COLORS.keys(), key=lambda p: party_totals.get(p, 0), reverse=True):
    total = party_totals.get(party, 0)
    speakers = len(party_speakers.get(party, set()))

    if total > 0:
        print(f"\n{party} ({total} Adjektive, {speakers} Redner):")
        print("-" * 50)
        for adj, count in party_adjectives[party].most_common(15):
            print(f"  {adj}: {count}")

# Vergleich: Wer verwendet welche negativen Adjektive?
print("\n" + "="*80)
print("VERGLEICH: SPEZIFISCH NEGATIVE ADJEKTIVE")
print("="*80)

negative_terms = ['illegal', 'kriminell', 'gefährlich', 'unkontrolliert',
                  'ausreisepflichtig', 'gescheitert', 'falsch', 'fremd']

print("\nAnzahl spezifisch negativer Adjektive pro Partei:")
print("-" * 50)
for party in sorted(PARTY_COLORS.keys(), key=lambda p: party_totals.get(p, 0), reverse=True):
    if party_totals.get(party, 0) > 0:
        neg_count = sum(party_adjectives[party].get(adj, 0) for adj in negative_terms)
        total = party_totals[party]
        pct = (neg_count / total * 100) if total > 0 else 0
        print(f"  {party}: {neg_count} ({pct:.1f}% von {total})")

# Speichere Ergebnisse
print("\n" + "="*80)
print("SPEICHERE ERGEBNISSE")
print("="*80)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# JSON Summary
summary = {
    'method': 'NER-basierte Redner-Extraktion',
    'total_segments': len(all_segments),
    'by_party': {}
}

for party in PARTY_COLORS.keys():
    if party in party_adjectives:
        summary['by_party'][party] = {
            'total_adjectives': party_totals.get(party, 0),
            'unique_speakers': len(party_speakers.get(party, set())),
            'top_adjectives': dict(party_adjectives[party].most_common(20)),
            'speakers': list(party_speakers.get(party, set()))[:10]
        }

with open(f'{OUTPUT_DIR}/adjectives_by_speaker_ner.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[OK] adjectives_by_speaker_ner.json gespeichert")

# CSV mit allen Segmenten
segments_df = pd.DataFrame(all_segments)
if len(segments_df) > 0:
    segments_df['adjectives_str'] = segments_df['adjectives'].apply(lambda x: ', '.join(x))
    segments_df.to_csv(f'{OUTPUT_DIR}/speaker_adjectives_migration.csv', index=False, encoding='utf-8')
    print(f"[OK] speaker_adjectives_migration.csv gespeichert")

# Visualisierungen
print("\n" + "="*80)
print("ERSTELLE VISUALISIERUNGEN")
print("="*80)

Path(VIZ_DIR).mkdir(parents=True, exist_ok=True)

# Plot 1: Anzahl Adjektive pro Partei (Redner-basiert)
fig, ax = plt.subplots(figsize=(10, 6))
parties = [p for p in PARTY_COLORS.keys() if party_totals.get(p, 0) > 0]
parties = sorted(parties, key=lambda p: party_totals[p], reverse=True)
counts = [party_totals[p] for p in parties]
colors = [PARTY_COLORS[p] for p in parties]

bars = ax.bar(parties, counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Partei')
ax.set_ylabel('Anzahl Adjektive')
ax.set_title('Adjektive im Migrations-Kontext nach Redner-Partei\n(NER-basierte Zuordnung)', fontsize=14, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/adjectives_by_speaker_party.png', dpi=300, bbox_inches='tight')
print(f"[OK] adjectives_by_speaker_party.png gespeichert")
plt.close()

# Plot 2: Anteil negativer Adjektive
fig, ax = plt.subplots(figsize=(10, 6))

neg_counts = []
for party in parties:
    neg = sum(party_adjectives[party].get(adj, 0) for adj in negative_terms)
    neg_counts.append(neg)

bars = ax.bar(parties, neg_counts, color=[PARTY_COLORS[p] for p in parties], edgecolor='black', alpha=0.8)
ax.set_xlabel('Partei')
ax.set_ylabel('Anzahl stark negativer Adjektive')
ax.set_title('Stark negative Adjektive im Migrations-Kontext nach Redner\n(illegal, kriminell, gefaehrlich, etc.)', fontsize=14, fontweight='bold')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/negative_adjectives_by_speaker.png', dpi=300, bbox_inches='tight')
print(f"[OK] negative_adjectives_by_speaker.png gespeichert")
plt.close()

# Plot 3: Heatmap der Top-Adjektive pro Partei
all_top = set()
for party in parties:
    for adj, _ in party_adjectives[party].most_common(10):
        all_top.add(adj)

all_top = sorted(list(all_top))

heatmap_data = []
for adj in all_top:
    row = [party_adjectives[p].get(adj, 0) for p in parties]
    heatmap_data.append(row)

heatmap_df = pd.DataFrame(heatmap_data, index=all_top, columns=parties)

fig, ax = plt.subplots(figsize=(12, max(8, len(all_top) * 0.4)))
sns.heatmap(heatmap_df, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Haeufigkeit'})
ax.set_xlabel('Partei')
ax.set_ylabel('Adjektiv')
ax.set_title('Adjektive im Migrations-Kontext: Wer verwendet was?\n(NER-basierte Redner-Zuordnung)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/adjectives_heatmap_by_speaker.png', dpi=300, bbox_inches='tight')
print(f"[OK] adjectives_heatmap_by_speaker.png gespeichert")
plt.close()

# Plot 4: Prozentuale Verteilung negativer Begriffe
fig, ax = plt.subplots(figsize=(10, 6))

percentages = []
for party in parties:
    total = party_totals[party]
    neg = sum(party_adjectives[party].get(adj, 0) for adj in negative_terms)
    pct = (neg / total * 100) if total > 0 else 0
    percentages.append(pct)

bars = ax.bar(parties, percentages, color=[PARTY_COLORS[p] for p in parties], edgecolor='black', alpha=0.8)
ax.set_xlabel('Partei')
ax.set_ylabel('Anteil (%)')
ax.set_title('Anteil stark negativer Adjektive am Gesamt\n(pro Redner-Partei)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(percentages) * 1.2 if percentages else 10)

for bar, pct in zip(bars, percentages):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VIZ_DIR}/negative_percentage_by_speaker.png', dpi=300, bbox_inches='tight')
print(f"[OK] negative_percentage_by_speaker.png gespeichert")
plt.close()

print("\n" + "="*80)
print("ANALYSE ABGESCHLOSSEN")
print("="*80)
print(f"\nErgebnisse in: {OUTPUT_DIR}")
print(f"Visualisierungen in: {VIZ_DIR}")
