"""
Verknuepfe Adjektive mit bereits vorhandenen BERT Hate-Speech Annotationen
Keine neue Klassifikation noetig - nutzt immigrant_docs_classified.parquet
"""
import pandas as pd
from collections import Counter
import json
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("="*70)
print("VERKNUEPFUNG: ADJEKTIVE MIT HATE-SPEECH ANNOTATIONEN")
print("="*70)

# 1. Lade bereits klassifizierte Dokumente
print("\nLade bereits klassifizierte Hate-Speech Daten...")
hate_df = pd.read_parquet(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\immigrant_docs_classified.parquet')
print(f"Geladene Dokumente: {len(hate_df)}")
print(f"Davon HATE: {len(hate_df[hate_df['hate_label']=='HATE'])}")
print(f"Davon NON_HATE: {len(hate_df[hate_df['hate_label']=='NON_HATE'])}")

# 2. Lade Adjektiv-Daten
print("\nLade Adjektiv-Daten...")
adj_df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_migration_context.csv')
print(f"Geladene Adjektive: {len(adj_df)}")

# 3. Verknuepfe ueber doc_id
print("\nVerknuepfe Daten ueber doc_id...")

# Erstelle Mapping: doc_id -> hate_label, hate_score
hate_mapping = hate_df.set_index('doc_id')[['hate_label', 'hate_score']].to_dict('index')

# Fuege Hate-Labels zu Adjektiven hinzu
adj_df['hate_label'] = adj_df['doc_id'].map(lambda x: hate_mapping.get(x, {}).get('hate_label', 'UNKNOWN'))
adj_df['hate_score'] = adj_df['doc_id'].map(lambda x: hate_mapping.get(x, {}).get('hate_score', 0))

# 4. Statistiken
print("\n" + "="*70)
print("ERGEBNISSE")
print("="*70)

print("\nAdjektive nach Hate-Label:")
print(adj_df['hate_label'].value_counts())

# Adjektive in HATE-Kontexten
hate_adj = adj_df[adj_df['hate_label'] == 'HATE']
non_hate_adj = adj_df[adj_df['hate_label'] == 'NON_HATE']

print(f"\nAdjektive in HATE-Dokumenten: {len(hate_adj)} ({len(hate_adj)/len(adj_df)*100:.2f}%)")
print(f"Adjektive in NON_HATE-Dokumenten: {len(non_hate_adj)} ({len(non_hate_adj)/len(adj_df)*100:.2f}%)")

# 5. Top Adjektive in HATE vs NON_HATE
print("\n" + "-"*50)
print("TOP 20 ADJEKTIVE IN HATE-DOKUMENTEN:")
print("-"*50)
hate_counts = Counter(hate_adj['adjective'].str.lower())
for adj, count in hate_counts.most_common(20):
    print(f"  {adj:25}: {count:4}x")

print("\n" + "-"*50)
print("TOP 20 ADJEKTIVE IN NON_HATE-DOKUMENTEN:")
print("-"*50)
non_hate_counts = Counter(non_hate_adj['adjective'].str.lower())
for adj, count in non_hate_counts.most_common(20):
    print(f"  {adj:25}: {count:4}x")

# 6. Vergleich: Welche Adjektive sind ueberproportional in HATE?
print("\n" + "="*70)
print("ADJEKTIVE MIT HOECHSTEM HATE-ANTEIL")
print("(Adjektive die ueberproportional in Hate-Speech vorkommen)")
print("="*70)

# Berechne Hate-Anteil pro Adjektiv
adj_stats = []
for adj in set(adj_df['adjective'].str.lower()):
    total = len(adj_df[adj_df['adjective'].str.lower() == adj])
    in_hate = len(hate_adj[hate_adj['adjective'].str.lower() == adj])
    if total >= 5:  # Mindestens 5 Vorkommen
        hate_ratio = in_hate / total * 100
        adj_stats.append({
            'adjective': adj,
            'total': total,
            'in_hate': in_hate,
            'hate_ratio': hate_ratio
        })

adj_stats_df = pd.DataFrame(adj_stats)
adj_stats_df = adj_stats_df.sort_values('hate_ratio', ascending=False)

print("\nAdjektive mit hoechstem Hate-Anteil (mind. 5 Vorkommen):")
print("-"*60)
print(f"{'Adjektiv':<25} {'Gesamt':>8} {'In HATE':>10} {'Anteil':>10}")
print("-"*60)
for _, row in adj_stats_df.head(25).iterrows():
    print(f"{row['adjective']:<25} {row['total']:>8} {row['in_hate']:>10} {row['hate_ratio']:>9.1f}%")

# 7. Speichern
output_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_with_hate_labels.csv'
adj_df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n[OK] Annotierte Adjektive gespeichert: {output_path}")

# Speichere nur HATE-Adjektive
hate_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_in_hate_documents.csv'
hate_adj.to_csv(hate_path, index=False, encoding='utf-8')
print(f"[OK] Adjektive in HATE-Dokumenten: {hate_path}")

# Speichere Statistiken
stats_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_hate_ratio_stats.csv'
adj_stats_df.to_csv(stats_path, index=False, encoding='utf-8')
print(f"[OK] Hate-Ratio Statistiken: {stats_path}")

# JSON Summary
summary = {
    "total_adjectives": len(adj_df),
    "adjectives_in_hate_docs": len(hate_adj),
    "adjectives_in_non_hate_docs": len(non_hate_adj),
    "hate_percentage": len(hate_adj)/len(adj_df)*100,
    "top_adjectives_in_hate": dict(hate_counts.most_common(30)),
    "top_adjectives_by_hate_ratio": adj_stats_df.head(20).to_dict('records')
}

summary_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_hate_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[OK] Zusammenfassung: {summary_path}")

print("\n[OK] Fertig!")
