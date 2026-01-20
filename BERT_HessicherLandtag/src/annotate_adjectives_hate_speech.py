"""
Annotiere Adjektive im Migrationskontext mit BERT Hate-Speech-Modell
Verwendet: Hate-speech-CNERG/dehatebert-mono-german
"""
import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("="*70)
print("BERT HATE-SPEECH ANNOTATION FUER ADJEKTIVE IM MIGRATIONSKONTEXT")
print("="*70)

# 1. GPU/CPU Check
if torch.cuda.is_available():
    device = 0
    print("[OK] CUDA GPU verfuegbar - verwende GPU")
else:
    device = -1
    print("[INFO] Keine GPU - verwende CPU (langsamer)")

# 2. Lade Hate-Speech Modell
print("\nLade Hate-Speech Modell (dehatebert-mono-german)...")
hate_pipe = pipeline(
    "text-classification",
    model="Hate-speech-CNERG/dehatebert-mono-german",
    device=device,
    batch_size=16,
    truncation=True
)
print("[OK] Modell geladen")

# 3. Lade die Adjektiv-Daten mit Kontext
print("\nLade Adjektiv-Daten...")
df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\negative_adjectives_migration.csv')
print(f"Geladene Datensaetze: {len(df)}")

# 4. Pruefe ob 'context' Spalte existiert
if 'context' not in df.columns:
    print("\n[WARNUNG] Keine 'context' Spalte gefunden. Lade Original-Daten...")
    df = pd.read_csv(r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_migration_context.csv')
    print(f"Geladene Datensaetze: {len(df)}")

print(f"Spalten: {df.columns.tolist()}")

# 5. Annotiere jeden Kontext mit Hate-Speech-Modell
print("\nAnnotiere Kontexte mit Hate-Speech-Modell...")

contexts = df['context'].fillna('').tolist()
batch_size = 16

results = []
for i in tqdm(range(0, len(contexts), batch_size), desc="Klassifiziere"):
    batch = contexts[i:i+batch_size]
    # Ersetze leere Strings mit Platzhalter
    batch = [t if t.strip() else "kein Text" for t in batch]
    batch_results = hate_pipe(batch)
    results.extend(batch_results)

# 6. Fuege Ergebnisse zum DataFrame hinzu
df['hate_label'] = [r['label'] for r in results]
df['hate_score'] = [r['score'] for r in results]

# 7. Statistiken
print("\n" + "="*70)
print("ERGEBNISSE")
print("="*70)

print("\nLabel-Verteilung:")
print(df['hate_label'].value_counts())

hate_contexts = df[df['hate_label'] == 'HATE']
print(f"\nKontexte mit HATE-Label: {len(hate_contexts)} ({len(hate_contexts)/len(df)*100:.2f}%)")

if len(hate_contexts) > 0:
    print(f"\nHate-Score Statistiken:")
    print(f"  Mittelwert: {hate_contexts['hate_score'].mean():.4f}")
    print(f"  Median:     {hate_contexts['hate_score'].median():.4f}")
    print(f"  Min:        {hate_contexts['hate_score'].min():.4f}")
    print(f"  Max:        {hate_contexts['hate_score'].max():.4f}")

    # Top Adjektive in Hate-Kontexten
    print("\nTop 20 Adjektive in HATE-Kontexten:")
    print("-"*40)
    from collections import Counter
    hate_adj = Counter(hate_contexts['adjective'].str.lower())
    for adj, count in hate_adj.most_common(20):
        print(f"  {adj:25}: {count:4}x")

    # Beispiele
    print("\nBeispiele von HATE-Kontexten (Top 5 nach Score):")
    print("="*70)
    top_hate = hate_contexts.nlargest(5, 'hate_score')
    for _, row in top_hate.iterrows():
        print(f"\nAdjektiv: {row['adjective']}")
        print(f"Hate-Score: {row['hate_score']:.4f}")
        context = row['context'][:200] + "..." if len(str(row['context'])) > 200 else row['context']
        print(f"Kontext: {context}")
        print("-"*50)

# 8. Speichern
output_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_hate_speech_annotated.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n[OK] Annotierte Daten gespeichert: {output_path}")

# Speichere nur HATE-Kontexte separat
if len(hate_contexts) > 0:
    hate_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_in_hate_contexts.csv'
    hate_contexts.to_csv(hate_path, index=False, encoding='utf-8')
    print(f"[OK] HATE-Kontexte gespeichert: {hate_path}")

# 9. Zusammenfassung als JSON
import json
summary = {
    "total_adjectives": len(df),
    "hate_contexts": len(hate_contexts),
    "hate_percentage": len(hate_contexts)/len(df)*100,
    "non_hate_contexts": len(df) - len(hate_contexts),
    "top_adjectives_in_hate": dict(hate_adj.most_common(30)) if len(hate_contexts) > 0 else {},
    "model": "Hate-speech-CNERG/dehatebert-mono-german"
}

summary_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\adjectives_hate_annotation_summary.json'
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[OK] Zusammenfassung gespeichert: {summary_path}")

print("\n[OK] Annotation abgeschlossen!")
