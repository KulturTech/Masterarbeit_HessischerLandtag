import pandas as pd
import re
from pathlib import Path

BASE_DIR = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")

# Lade gefilterte HATE-Reden
df = pd.read_csv(BASE_DIR / 'Data/evaluation/hate_speeches_for_lime.csv')

def extract_speech_content(text):
    """Entfernt Session-Header und TOC-Artefakte."""
    # Session-Header: "Hessischer Landtag ... Sitzung ... [Datum]"
    text = re.sub(r'^Hessischer Landtag.*?\d{1,2}\.\s*\w+\s*\d{4}\s*\n+', '', text, flags=re.DOTALL)

    # Seitenzahlen am Anfang
    text = re.sub(r'^\d+\s*\n+', '', text)

    # Inhaltsverzeichnis-Artefakte entfernen: "Name.......123" oder "7. \nAntrag"
    text = re.sub(r'^(\d+\.\s*\n+)+', '', text)
    text = re.sub(r'\n+\d+\s*\n+', '\n', text)

    # Zu viele Punkte entfernen (Dot-Leader)
    text = re.sub(r'\.{5,}', '... ', text)

    # Mehrfache Zeilenumbrüche normalisieren
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

df['text_clean'] = df['text'].apply(extract_speech_content)

print("Top-5 bereinigte HATE-Texte:")
for i, row in df.head(5).iterrows():
    print(f"\n{i+1}. Score: {row['score']:.4f}")
    print(f"   Länge (roh): {len(row['text'])}, (bereinigt): {len(row['text_clean'])}")
    print(f"   Preview: {row['text_clean'][:150]}...")

# Speichern für LIME
df.to_csv(BASE_DIR / 'Data/evaluation/hate_speeches_for_lime_cleaned.csv', index=False)
print(f"\n✓ Gespeichert (mit bereinigten Texten): hate_speeches_for_lime_cleaned.csv")
