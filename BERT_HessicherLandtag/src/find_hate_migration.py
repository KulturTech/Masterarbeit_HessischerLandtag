import pandas as pd, re, sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def max_sent(text):
    parts = re.split(r'(?<=[.!?])\s+', str(text))
    return max((len(s.split()) for s in parts), default=0)

df = pd.read_csv(
    r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\labeled_data_all.csv'
)
keywords = ['Migrant','Flüchtling','Asyl','Einwanderer','Ausländer','Migration','Zuwanderer','Abschiebung']
mask = df['label']=='HATE'
mig  = df['text'].str.contains('|'.join(keywords), case=False, na=False)
subset = df[mask & mig].copy()
subset['max_sent'] = subset['text'].apply(max_sent)
clean = subset[subset['max_sent'] >= 15].sort_values('max_sent', ascending=False)
print(f'Echte Redetexte HATE+Migration: {len(clean)}')
out = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\evaluation\training_hate_migration.csv'
clean.to_csv(out, index=False, encoding='utf-8')
for _, row in clean.head(3).iterrows():
    print(f'\n--- max_sent={row["max_sent"]} ---')
    print(row['text'][:300])
