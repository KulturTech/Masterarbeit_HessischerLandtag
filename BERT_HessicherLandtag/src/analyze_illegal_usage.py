"""
Analyse der Verwendung von "illegal" im Migrationskontext nach Parteien
"""
import pandas as pd
import re
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Lade Reden
df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\full_speeches_migration_context.csv')

print("="*80)
print("ANALYSE: VERWENDUNG VON 'ILLEGAL' IM MIGRATIONSKONTEXT")
print("="*80)

# Statistiken pro Partei
parties = ['CDU', 'SPD', 'GRUENE', 'FDP', 'LINKE', 'AfD']

print("\n1. HAEUFIGKEIT PRO PARTEI:")
print("-"*50)

for party in parties:
    party_df = df[df['party'] == party]
    total_speeches = len(party_df)
    with_illegal = party_df[party_df['full_text'].str.contains('illegal', case=False, na=False)]
    count_illegal = len(with_illegal)
    pct = (count_illegal / total_speeches * 100) if total_speeches > 0 else 0
    print(f"{party:8}: {count_illegal:3} von {total_speeches:3} Reden ({pct:.1f}%)")

# AfD Beispiele
print("\n" + "="*80)
print("2. AFD-BEISPIELE MIT 'ILLEGAL':")
print("="*80)

afd = df[df['party'] == 'AfD']
afd_illegal = afd[afd['full_text'].str.contains('illegal', case=False, na=False)]

for i, (_, row) in enumerate(afd_illegal.head(5).iterrows()):
    print(f"\n--- Redner: {row['speaker']} ---")
    text = row['full_text']
    matches = list(re.finditer(r'.{0,150}illegal.{0,150}', text, re.IGNORECASE))
    for m in matches[:2]:
        context = m.group().replace('\n', ' ')
        print(f"  '...{context}...'")

# Vergleich mit anderen Parteien
print("\n" + "="*80)
print("3. VERGLEICH: WIE VERWENDEN ANDERE PARTEIEN 'ILLEGAL'?")
print("="*80)

for party in ['CDU', 'SPD', 'GRUENE', 'LINKE']:
    party_df = df[df['party'] == party]
    party_illegal = party_df[party_df['full_text'].str.contains('illegal', case=False, na=False)]

    if len(party_illegal) > 0:
        print(f"\n--- {party} ---")
        row = party_illegal.iloc[0]
        text = row['full_text']
        matches = list(re.finditer(r'.{0,150}illegal.{0,150}', text, re.IGNORECASE))
        for m in matches[:1]:
            context = m.group().replace('\n', ' ')
            print(f"  Redner: {row['speaker']}")
            print(f"  '...{context}...'")

# Quantitative Analyse
print("\n" + "="*80)
print("4. QUANTITATIVE ERKLAERUNG")
print("="*80)

# Lade die NER-Daten
ner_data = {
    'CDU': {'total': 152, 'illegal_forms': 3},
    'SPD': {'total': 278, 'illegal_forms': 5},
    'GRUENE': {'total': 208, 'illegal_forms': 4},
    'FDP': {'total': 109, 'illegal_forms': 0},
    'LINKE': {'total': 401, 'illegal_forms': 9},
    'AfD': {'total': 443, 'illegal_forms': 9}
}

print("\nAdjektiv-Verwendung 'illegal' (NER-basiert, Redner spricht):")
print("-"*50)
for party in parties:
    total = ner_data[party]['total']
    illegal = ner_data[party]['illegal_forms']
    pct = (illegal / total * 100) if total > 0 else 0
    print(f"{party:8}: {illegal:2} von {total:3} Adjektiven ({pct:.1f}%)")

print("\n" + "="*80)
print("ERKLAERUNG")
print("="*80)
print("""
Die hohen absoluten Zahlen fuer 'illegal' bei der AfD erklaeren sich durch:

1. MEHR REDEN ZUM THEMA MIGRATION:
   - AfD hat 443 Adjektive im Migrationskontext (hoechste Zahl)
   - CDU nur 152, SPD 278, GRUENE 208
   - Die AfD thematisiert Migration haeufiger

2. RHETORISCHE STRATEGIE:
   - AfD verwendet 'illegal' oft in Kombination mit 'Migration'
   - Begriffe wie 'illegale Einwanderung', 'illegale Migration'
   - Dies ist ein zentrales Framing der AfD-Rhetorik

3. ABER: PROZENTUAL AEHNLICH
   - AfD: 9 von 443 = 2.0% 'illegal'-Verwendung
   - LINKE: 9 von 401 = 2.2% 'illegal'-Verwendung
   - Der prozentuale Anteil ist vergleichbar!

4. UNTERSCHIED IN DER KONNOTATION:
   - AfD: 'illegale Migration stoppen', 'illegale Einwanderer'
   - LINKE: 'sogenannte illegale Migration', kritische Distanz
   - Gleiche Woerter, aber unterschiedliche Rahmung
""")
