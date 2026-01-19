"""
Analyse negativer Adjektive im Kontext von Migrant*innen
========================================================
Dieses Skript identifiziert und analysiert negative Adjektive,
die in Verbindung mit Migrations-bezogenen Begriffen verwendet werden.

Ansatz: BERT-basierte Sentiment-Analyse + Wörterbuch-basierte Adjektiv-Erkennung
"""

import torch
from transformers import pipeline
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Konfiguration
CONFIG = {
    'batch_size': 8,
    'context_window': 50,  # Wörter vor/nach dem Migrations-Begriff
    'min_adjective_freq': 3,  # Mindesthäufigkeit für Adjektive
    'sentiment_threshold': 0.6,  # Schwellenwert für negative Klassifikation
}

print("="*80)
print("ANALYSE NEGATIVER ADJEKTIVE IM KONTEXT VON MIGRANT*INNEN")
print("="*80)

# 1. GPU/CPU Setup
if torch.cuda.is_available():
    device = 0
    print("[OK] CUDA-GPU verfuegbar")
else:
    device = -1
    print("[OK] CPU-Modus")

# 2. Deutsche Adjektive (häufige + politisch relevante)
# Basiert auf deutschen Wortlisten und politischem Diskurs
GERMAN_ADJECTIVES = {
    # Negative Adjektive (häufig im politischen Diskurs)
    'illegal', 'illegalen', 'illegaler', 'illegale',
    'kriminell', 'kriminelle', 'kriminellen', 'krimineller',
    'gefährlich', 'gefährliche', 'gefährlichen', 'gefährlicher',
    'unkontrolliert', 'unkontrollierte', 'unkontrollierten',
    'massenhaft', 'massenhafte', 'massenhaften',
    'ungeregelt', 'ungeregelte', 'ungeregelten',
    'problematisch', 'problematische', 'problematischen',
    'schwierig', 'schwierige', 'schwierigen',
    'bedrohlich', 'bedrohliche', 'bedrohlichen',
    'fremd', 'fremde', 'fremden', 'fremder',
    'unerwünscht', 'unerwünschte', 'unerwünschten',
    'abgelehnt', 'abgelehnte', 'abgelehnten',
    'straffällig', 'straffällige', 'straffälligen',
    'gewalttätig', 'gewalttätige', 'gewalttätigen',
    'radikal', 'radikale', 'radikalen', 'radikaler',
    'extremistisch', 'extremistische', 'extremistischen',
    'islamistisch', 'islamistische', 'islamistischen',
    'unintegrier', 'unintegriert', 'unintegrierte',
    'arbeitslos', 'arbeitslose', 'arbeitslosen',
    'unqualifiziert', 'unqualifizierte', 'unqualifizierten',
    'belastend', 'belastende', 'belastenden',
    'teuer', 'teure', 'teuren', 'teurer',
    'kostspielig', 'kostspielige', 'kostspieligen',
    'falsch', 'falsche', 'falschen',
    'gescheitert', 'gescheiterte', 'gescheiterten',
    'fehlgeschlagen', 'fehlgeschlagene', 'fehlgeschlagenen',
    'abzuschieben', 'abzuschiebende', 'abzuschiebenden',
    'ausreisepflichtig', 'ausreisepflichtige', 'ausreisepflichtigen',

    # Positive/Neutrale Adjektive (zum Vergleich)
    'legal', 'legale', 'legalen', 'legaler',
    'qualifiziert', 'qualifizierte', 'qualifizierten',
    'integriert', 'integrierte', 'integrierten',
    'erfolgreich', 'erfolgreiche', 'erfolgreichen',
    'fleißig', 'fleißige', 'fleißigen',
    'berechtigt', 'berechtigte', 'berechtigten',
    'anerkannt', 'anerkannte', 'anerkannten',
    'schutzbedürftig', 'schutzbedürftige', 'schutzbedürftigen',
    'verfolgt', 'verfolgte', 'verfolgten',
    'humanitär', 'humanitäre', 'humanitären',
    'willkommen', 'willkommene', 'willkommenen',
    'notwendig', 'notwendige', 'notwendigen',
    'wichtig', 'wichtige', 'wichtigen', 'wichtiger',
    'gut', 'gute', 'guten', 'guter',
    'positiv', 'positive', 'positiven',
    'wertvoll', 'wertvolle', 'wertvollen',
    'sicher', 'sichere', 'sicheren', 'sicherer',
    'geordnet', 'geordnete', 'geordneten',
    'kontrolliert', 'kontrollierte', 'kontrollierten',
    'regulär', 'reguläre', 'regulären',
    'gesteuert', 'gesteuerte', 'gesteuerten',

    # Allgemeine deskriptive Adjektive
    'hoch', 'hohe', 'hohen', 'hoher',
    'niedrig', 'niedrige', 'niedrigen',
    'groß', 'große', 'großen', 'großer',
    'klein', 'kleine', 'kleinen', 'kleiner',
    'viel', 'viele', 'vielen', 'vieler',
    'wenig', 'wenige', 'wenigen',
    'stark', 'starke', 'starken', 'starker',
    'schwach', 'schwache', 'schwachen',
    'neu', 'neue', 'neuen', 'neuer',
    'alt', 'alte', 'alten', 'alter',
    'jung', 'junge', 'jungen', 'junger',
    'schnell', 'schnelle', 'schnellen',
    'langsam', 'langsame', 'langsamen',
    'kurz', 'kurze', 'kurzen', 'kurzer',
    'lang', 'lange', 'langen', 'langer',
    'aktuell', 'aktuelle', 'aktuellen',
    'zusätzlich', 'zusätzliche', 'zusätzlichen',
    'bestehend', 'bestehende', 'bestehenden',
    'zunehmend', 'zunehmende', 'zunehmenden',
    'steigend', 'steigende', 'steigenden',
    'sinkend', 'sinkende', 'sinkenden',
    'wachsend', 'wachsende', 'wachsenden',
    'europäisch', 'europäische', 'europäischen',
    'deutsch', 'deutsche', 'deutschen', 'deutscher',
    'ausländisch', 'ausländische', 'ausländischen',
    'international', 'internationale', 'internationalen',
    'national', 'nationale', 'nationalen',
    'politisch', 'politische', 'politischen',
    'sozial', 'soziale', 'sozialen',
    'wirtschaftlich', 'wirtschaftliche', 'wirtschaftlichen',
    'gesellschaftlich', 'gesellschaftliche', 'gesellschaftlichen',
}

# Lemma-Mapping für Normalisierung
ADJECTIVE_LEMMAS = {
    'illegalen': 'illegal', 'illegaler': 'illegal', 'illegale': 'illegal',
    'kriminelle': 'kriminell', 'kriminellen': 'kriminell', 'krimineller': 'kriminell',
    'gefährliche': 'gefährlich', 'gefährlichen': 'gefährlich', 'gefährlicher': 'gefährlich',
    'unkontrollierte': 'unkontrolliert', 'unkontrollierten': 'unkontrolliert',
    'massenhafte': 'massenhaft', 'massenhaften': 'massenhaft',
    'ungeregelte': 'ungeregelt', 'ungeregelten': 'ungeregelt',
    'problematische': 'problematisch', 'problematischen': 'problematisch',
    'schwierige': 'schwierig', 'schwierigen': 'schwierig',
    'bedrohliche': 'bedrohlich', 'bedrohlichen': 'bedrohlich',
    'fremde': 'fremd', 'fremden': 'fremd', 'fremder': 'fremd',
    'unerwünschte': 'unerwünscht', 'unerwünschten': 'unerwünscht',
    'abgelehnte': 'abgelehnt', 'abgelehnten': 'abgelehnt',
    'straffällige': 'straffällig', 'straffälligen': 'straffällig',
    'gewalttätige': 'gewalttätig', 'gewalttätigen': 'gewalttätig',
    'radikale': 'radikal', 'radikalen': 'radikal', 'radikaler': 'radikal',
    'extremistische': 'extremistisch', 'extremistischen': 'extremistisch',
    'islamistische': 'islamistisch', 'islamistischen': 'islamistisch',
    'legale': 'legal', 'legalen': 'legal', 'legaler': 'legal',
    'qualifizierte': 'qualifiziert', 'qualifizierten': 'qualifiziert',
    'integrierte': 'integriert', 'integrierten': 'integriert',
    'erfolgreiche': 'erfolgreich', 'erfolgreichen': 'erfolgreich',
    'fleißige': 'fleißig', 'fleißigen': 'fleißig',
    'berechtigte': 'berechtigt', 'berechtigten': 'berechtigt',
    'anerkannte': 'anerkannt', 'anerkannten': 'anerkannt',
    'schutzbedürftige': 'schutzbedürftig', 'schutzbedürftigen': 'schutzbedürftig',
    'verfolgte': 'verfolgt', 'verfolgten': 'verfolgt',
    'humanitäre': 'humanitär', 'humanitären': 'humanitär',
    'willkommene': 'willkommen', 'willkommenen': 'willkommen',
    'notwendige': 'notwendig', 'notwendigen': 'notwendig',
    'wichtige': 'wichtig', 'wichtigen': 'wichtig', 'wichtiger': 'wichtig',
    'gute': 'gut', 'guten': 'gut', 'guter': 'gut',
    'positive': 'positiv', 'positiven': 'positiv',
    'wertvolle': 'wertvoll', 'wertvollen': 'wertvoll',
    'sichere': 'sicher', 'sicheren': 'sicher', 'sicherer': 'sicher',
    'geordnete': 'geordnet', 'geordneten': 'geordnet',
    'kontrollierte': 'kontrolliert', 'kontrollierten': 'kontrolliert',
    'reguläre': 'regulär', 'regulären': 'regulär',
    'gesteuerte': 'gesteuert', 'gesteuerten': 'gesteuert',
    'hohe': 'hoch', 'hohen': 'hoch', 'hoher': 'hoch',
    'niedrige': 'niedrig', 'niedrigen': 'niedrig',
    'große': 'groß', 'großen': 'groß', 'großer': 'groß',
    'kleine': 'klein', 'kleinen': 'klein', 'kleiner': 'klein',
    'viele': 'viel', 'vielen': 'viel', 'vieler': 'viel',
    'wenige': 'wenig', 'wenigen': 'wenig',
    'starke': 'stark', 'starken': 'stark', 'starker': 'stark',
    'schwache': 'schwach', 'schwachen': 'schwach',
    'neue': 'neu', 'neuen': 'neu', 'neuer': 'neu',
    'alte': 'alt', 'alten': 'alt', 'alter': 'alt',
    'junge': 'jung', 'jungen': 'jung', 'junger': 'jung',
    'schnelle': 'schnell', 'schnellen': 'schnell',
    'langsame': 'langsam', 'langsamen': 'langsam',
    'kurze': 'kurz', 'kurzen': 'kurz', 'kurzer': 'kurz',
    'lange': 'lang', 'langen': 'lang', 'langer': 'lang',
    'aktuelle': 'aktuell', 'aktuellen': 'aktuell',
    'zusätzliche': 'zusätzlich', 'zusätzlichen': 'zusätzlich',
    'bestehende': 'bestehend', 'bestehenden': 'bestehend',
    'zunehmende': 'zunehmend', 'zunehmenden': 'zunehmend',
    'steigende': 'steigend', 'steigenden': 'steigend',
    'sinkende': 'sinkend', 'sinkenden': 'sinkend',
    'wachsende': 'wachsend', 'wachsenden': 'wachsend',
    'europäische': 'europäisch', 'europäischen': 'europäisch',
    'deutsche': 'deutsch', 'deutschen': 'deutsch', 'deutscher': 'deutsch',
    'ausländische': 'ausländisch', 'ausländischen': 'ausländisch',
    'internationale': 'international', 'internationalen': 'international',
    'nationale': 'national', 'nationalen': 'national',
    'politische': 'politisch', 'politischen': 'politisch',
    'soziale': 'sozial', 'sozialen': 'sozial',
    'wirtschaftliche': 'wirtschaftlich', 'wirtschaftlichen': 'wirtschaftlich',
    'gesellschaftliche': 'gesellschaftlich', 'gesellschaftlichen': 'gesellschaftlich',
    'arbeitslose': 'arbeitslos', 'arbeitslosen': 'arbeitslos',
    'unqualifizierte': 'unqualifiziert', 'unqualifizierten': 'unqualifiziert',
    'belastende': 'belastend', 'belastenden': 'belastend',
    'teure': 'teuer', 'teuren': 'teuer', 'teurer': 'teuer',
    'kostspielige': 'kostspielig', 'kostspieligen': 'kostspielig',
    'falsche': 'falsch', 'falschen': 'falsch',
    'gescheiterte': 'gescheitert', 'gescheiterten': 'gescheitert',
    'fehlgeschlagene': 'fehlgeschlagen', 'fehlgeschlagenen': 'fehlgeschlagen',
    'abzuschiebende': 'abzuschieben', 'abzuschiebenden': 'abzuschieben',
    'ausreisepflichtige': 'ausreisepflichtig', 'ausreisepflichtigen': 'ausreisepflichtig',
}

def get_lemma(word):
    """Gibt das Lemma eines Adjektivs zurück"""
    word_lower = word.lower()
    return ADJECTIVE_LEMMAS.get(word_lower, word_lower)

print("[OK] Adjektiv-Woerterbuch geladen ({} Formen)".format(len(GERMAN_ADJECTIVES)))

# 3. Lade BERT-Sentiment-Modell (deutsches Sentiment-Modell)
print("\nLade BERT Sentiment-Modell...")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="oliverguhr/german-sentiment-bert",
    device=device,
    truncation=True,
    max_length=512
)
print("[OK] Sentiment-Modell geladen")

# 4. Migrations-Keywords (erweitert)
MIGRATION_KEYWORDS = [
    # Personen
    'migrant', 'migranten', 'migrantin', 'migrantinnen',
    'immigrant', 'immigranten', 'immigrantin', 'immigrantinnen',
    'einwanderer', 'einwanderin', 'einwanderinnen',
    'zuwanderer', 'zuwanderin', 'zuwanderinnen',
    'flüchtling', 'flüchtlinge', 'geflüchtete', 'geflüchteten',
    'asylbewerber', 'asylbewerberin', 'asylsuchende', 'asylsuchenden',
    'ausländer', 'ausländerin', 'ausländerinnen',
    'schutzsuchende', 'schutzsuchenden',
    # Abstrakte Begriffe
    'migration', 'zuwanderung', 'einwanderung', 'flucht', 'asyl',
]

# Regex-Pattern für Migrations-Begriffe
migration_pattern = re.compile(
    r'\b(' + '|'.join(MIGRATION_KEYWORDS) + r')\w*\b',
    re.IGNORECASE
)

def extract_context_around_migration_terms(text, window=50):
    """
    Extrahiert Textabschnitte rund um Migrations-Begriffe
    """
    if not isinstance(text, str):
        return []

    contexts = []
    words = text.split()

    for i, word in enumerate(words):
        if migration_pattern.search(word):
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            context = ' '.join(words[start:end])
            contexts.append({
                'keyword': word.lower(),
                'context': context,
                'position': i
            })

    return contexts

def extract_adjectives_from_context(context_text):
    """
    Extrahiert alle Adjektive aus dem Kontext mittels Wörterbuch-Matching
    """
    adjectives = []
    words = context_text.split()

    for i, word in enumerate(words):
        # Entferne Satzzeichen
        clean_word = re.sub(r'[^\w\-äöüßÄÖÜ]', '', word).lower()

        if clean_word in GERMAN_ADJECTIVES:
            adjectives.append({
                'adjective': clean_word,
                'lemma': get_lemma(clean_word),
                'position': i
            })

    return adjectives

def analyze_sentiment_of_context(context_text):
    """
    Analysiert das Sentiment des Kontexts mit BERT
    """
    try:
        result = sentiment_pipe(context_text[:512])[0]  # Max 512 Tokens
        return {
            'label': result['label'],
            'score': result['score']
        }
    except Exception as e:
        return {'label': 'neutral', 'score': 0.5}

def classify_adjective_sentiment(adjective, context):
    """
    Klassifiziert das Sentiment eines spezifischen Adjektivs im Kontext
    """
    # Erstelle einen fokussierten Satz mit dem Adjektiv
    focused_text = f"Das ist {adjective}."

    try:
        result = sentiment_pipe(focused_text)[0]
        return result
    except:
        return {'label': 'neutral', 'score': 0.5}

# 5. Lade Daten
print("\n" + "="*80)
print("LADE DATEN")
print("="*80)

data_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_clean.parquet'
print(f"Lade: {data_path}")
df = pd.read_parquet(data_path)
print(f"[OK] {len(df):,} Dokumente geladen")

# 6. Filtere Dokumente mit Migrations-Bezug
print("\n" + "="*80)
print("FILTERE MIGRATIONS-RELEVANTE DOKUMENTE")
print("="*80)

df['has_migration_ref'] = df['text'].apply(lambda x: bool(migration_pattern.search(str(x))))
migration_docs = df[df['has_migration_ref']].copy()
print(f"[OK] {len(migration_docs):,} Dokumente mit Migrations-Bezug ({len(migration_docs)/len(df)*100:.2f}%)")

# 7. Hauptanalyse
print("\n" + "="*80)
print("ANALYSIERE ADJEKTIVE IM MIGRATIONS-KONTEXT")
print("="*80)

all_adjectives = []
adjective_contexts = defaultdict(list)
negative_adjectives = Counter()
positive_adjectives = Counter()
neutral_adjectives = Counter()

# Analysiere jeden Text
for idx, row in tqdm(migration_docs.iterrows(), total=len(migration_docs), desc="Analysiere Dokumente"):
    text = row['text']
    doc_id = row.get('doc_id', idx)

    # Extrahiere Kontexte um Migrations-Begriffe
    contexts = extract_context_around_migration_terms(text, window=CONFIG['context_window'])

    for ctx in contexts:
        # Extrahiere Adjektive aus dem Kontext
        adjectives = extract_adjectives_from_context(ctx['context'])

        if adjectives:
            # Analysiere Sentiment des gesamten Kontexts
            context_sentiment = analyze_sentiment_of_context(ctx['context'])

            for adj in adjectives:
                adj_data = {
                    'doc_id': doc_id,
                    'keyword': ctx['keyword'],
                    'adjective': adj['adjective'],
                    'lemma': adj['lemma'],
                    'context': ctx['context'],
                    'context_sentiment': context_sentiment['label'],
                    'context_score': context_sentiment['score']
                }

                all_adjectives.append(adj_data)
                adjective_contexts[adj['lemma']].append(adj_data)

                # Klassifiziere nach Sentiment
                if context_sentiment['label'] == 'negative':
                    negative_adjectives[adj['lemma']] += 1
                elif context_sentiment['label'] == 'positive':
                    positive_adjectives[adj['lemma']] += 1
                else:
                    neutral_adjectives[adj['lemma']] += 1

# 8. Erstelle Ergebnis-DataFrames
print("\n" + "="*80)
print("ERGEBNISSE")
print("="*80)

adjectives_df = pd.DataFrame(all_adjectives)
print(f"\n[OK] {len(adjectives_df):,} Adjektiv-Vorkommen gefunden")
print(f"[OK] {len(set(adjectives_df['lemma'])):,} unique Adjektive")

# Top negative Adjektive
print("\n" + "-"*40)
print("TOP 30 NEGATIVE ADJEKTIVE (in negativem Kontext):")
print("-"*40)
for adj, count in negative_adjectives.most_common(30):
    if count >= CONFIG['min_adjective_freq']:
        print(f"  {adj}: {count}")

# Top positive Adjektive (zum Vergleich)
print("\n" + "-"*40)
print("TOP 20 POSITIVE ADJEKTIVE (zum Vergleich):")
print("-"*40)
for adj, count in positive_adjectives.most_common(20):
    if count >= CONFIG['min_adjective_freq']:
        print(f"  {adj}: {count}")

# 9. Statistiken
print("\n" + "="*80)
print("STATISTIKEN")
print("="*80)

total_negative = sum(negative_adjectives.values())
total_positive = sum(positive_adjectives.values())
total_neutral = sum(neutral_adjectives.values())
total_all = total_negative + total_positive + total_neutral

print(f"\nSentiment-Verteilung der Adjektiv-Kontexte:")
print(f"  Negativ:  {total_negative:,} ({total_negative/total_all*100:.1f}%)")
print(f"  Positiv:  {total_positive:,} ({total_positive/total_all*100:.1f}%)")
print(f"  Neutral:  {total_neutral:,} ({total_neutral/total_all*100:.1f}%)")

# 10. Speichere Ergebnisse
print("\n" + "="*80)
print("SPEICHERE ERGEBNISSE")
print("="*80)

output_dir = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1'

# Alle Adjektive mit Kontext
adjectives_df.to_parquet(f'{output_dir}/adjectives_migration_context.parquet')
adjectives_df.to_csv(f'{output_dir}/adjectives_migration_context.csv', index=False, encoding='utf-8')
print(f"[OK] Adjektive gespeichert: adjectives_migration_context.parquet/.csv")

# Nur negative Adjektive
negative_df = adjectives_df[adjectives_df['context_sentiment'] == 'negative']
negative_df.to_parquet(f'{output_dir}/negative_adjectives_migration.parquet')
negative_df.to_csv(f'{output_dir}/negative_adjectives_migration.csv', index=False, encoding='utf-8')
print(f"[OK] Negative Adjektive gespeichert: negative_adjectives_migration.parquet/.csv")

# Zusammenfassung als JSON
summary = {
    'timestamp': datetime.now().isoformat(),
    'total_documents': len(df),
    'migration_documents': len(migration_docs),
    'total_adjective_occurrences': len(adjectives_df),
    'unique_adjectives': len(set(adjectives_df['lemma'])),
    'sentiment_distribution': {
        'negative': total_negative,
        'positive': total_positive,
        'neutral': total_neutral
    },
    'top_30_negative_adjectives': dict(negative_adjectives.most_common(30)),
    'top_20_positive_adjectives': dict(positive_adjectives.most_common(20)),
}

with open(f'{output_dir}/adjective_analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[OK] Zusammenfassung gespeichert: adjective_analysis_summary.json")

# 11. Visualisierungen
print("\n" + "="*80)
print("ERSTELLE VISUALISIERUNGEN")
print("="*80)

viz_dir = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\visualizations'

# Plot 1: Top 20 negative Adjektive
fig, ax = plt.subplots(figsize=(12, 8))
top_neg = dict(negative_adjectives.most_common(20))
bars = ax.barh(list(top_neg.keys())[::-1], list(top_neg.values())[::-1], color='crimson')
ax.set_xlabel('Häufigkeit')
ax.set_title('Top 20 Adjektive in negativem Kontext\n(bei Migrations-Begriffen)', fontsize=14)
ax.bar_label(bars, padding=3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/top_negative_adjectives_migration.png', dpi=300, bbox_inches='tight')
print(f"[OK] Plot gespeichert: top_negative_adjectives_migration.png")

# Plot 2: Sentiment-Verteilung (Pie Chart)
fig, ax = plt.subplots(figsize=(8, 8))
sizes = [total_negative, total_positive, total_neutral]
labels = [f'Negativ\n({total_negative:,})', f'Positiv\n({total_positive:,})', f'Neutral\n({total_neutral:,})']
colors = ['#dc3545', '#28a745', '#6c757d']
explode = (0.05, 0, 0)
ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
       shadow=True, startangle=90)
ax.set_title('Sentiment-Verteilung der Adjektiv-Kontexte\n(Migrations-bezogene Texte)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{viz_dir}/sentiment_distribution_adjectives.png', dpi=300, bbox_inches='tight')
print(f"[OK] Plot gespeichert: sentiment_distribution_adjectives.png")

# Plot 3: Vergleich negative vs. positive Adjektive
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Negative
top_neg_20 = dict(negative_adjectives.most_common(15))
axes[0].barh(list(top_neg_20.keys())[::-1], list(top_neg_20.values())[::-1], color='crimson')
axes[0].set_xlabel('Häufigkeit')
axes[0].set_title('Top 15 Adjektive\n(negativer Kontext)', fontsize=12)

# Positive
top_pos_20 = dict(positive_adjectives.most_common(15))
axes[1].barh(list(top_pos_20.keys())[::-1], list(top_pos_20.values())[::-1], color='forestgreen')
axes[1].set_xlabel('Häufigkeit')
axes[1].set_title('Top 15 Adjektive\n(positiver Kontext)', fontsize=12)

plt.suptitle('Adjektive im Migrations-Kontext: Negativ vs. Positiv', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{viz_dir}/negative_vs_positive_adjectives.png', dpi=300, bbox_inches='tight')
print(f"[OK] Plot gespeichert: negative_vs_positive_adjectives.png")

plt.close('all')

print("\n" + "="*80)
print("ANALYSE ABGESCHLOSSEN")
print("="*80)
print(f"\nErgebnisse in: {output_dir}")
print(f"Visualisierungen in: {viz_dir}")
