"""
Annotationstool für Migrations-Hate-Speech
===========================================
Präsentiert Redeabschnitte aus dem Hessischen Landtag und fragt nach einer
Hate-Speech-Einschätzung.

Definition HATE (für diesen Kontext):
  Text enthält abwertende, entmenschlichende oder hetzerische Aussagen
  gegenüber Migranten, Flüchtlingen oder Ausländern als Gruppe.
  Dazu zählt: Dehumanisierung, Kriminalisierung als Gruppe, Aufrufe zur
  Ausgrenzung, explizite Feindseligkeit.

  NICHT als HATE zählt: sachliche Kritik an Migrationspolitik, Statistiken,
  neutrale Beschreibungen, Forderungen nach Regeländerungen ohne Feindseligkeit.

Steuerung:
  h  → HATE
  n  → NON_HATE
  s  → Überspringen (unsicher, später)
  q  → Speichern & Beenden

Fortschritt wird alle 10 Texte automatisch gespeichert.
"""

import os
import sys
from pathlib import Path

import pandas as pd

INPUT_FILE = Path(__file__).parent / 'annotation_sample.csv'

if not INPUT_FILE.exists():
    print(f"FEHLER: {INPUT_FILE} nicht gefunden.")
    print("Bitte zuerst ausführen: python src/prepare_annotation_sample.py")
    sys.exit(1)

df = pd.read_csv(INPUT_FILE)

if 'label' not in df.columns:
    df['label'] = ''

# Fortschritt: überspringe bereits kodierte Zeilen
bereits_kodiert = df['label'].isin(['HATE', 'NON_HATE'])
start_idx = bereits_kodiert.sum()

print("\n" + "="*70)
print("  HATE-SPEECH ANNOTATION — HESSISCHER LANDTAG")
print("="*70)
print(f"  Gesamt: {len(df)} Texte  |  Bereits kodiert: {start_idx}")
print()
print("  DEFINITION HATE:")
print("    Abwertende/entmenschlichende Aussagen über Migranten, Flüchtlinge")
print("    oder Ausländer als Gruppe.")
print("    NICHT HATE: sachliche Politikkritik, Statistiken, neutrale Berichte.")
print()
print("  Tasten:  h = HATE   n = NON_HATE   s = Überspringen   q = Beenden")
print("="*70)
input("\n  Enter drücken zum Starten...")

for i in range(start_idx, len(df)):
    row = df.iloc[i]

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n[{i+1}/{len(df)}]  (kodiert: {i}  |  p_hate={row.get('p_hate', '?'):.3f})\n")
    print("-"*70)

    text = str(row['text'])
    if len(text) > 900:
        print(text[:900])
        print(f"\n... [Text gekürzt, {len(text)} Zeichen gesamt]")
    else:
        print(text)

    print("-"*70)
    print("\n  [h] HATE    [n] NON_HATE    [s] Überspringen    [q] Beenden\n")

    while True:
        taste = input("  Eingabe: ").strip().lower()
        if taste == 'h':
            df.at[i, 'label'] = 'HATE'
            break
        elif taste == 'n':
            df.at[i, 'label'] = 'NON_HATE'
            break
        elif taste == 's':
            break
        elif taste == 'q':
            df.to_csv(INPUT_FILE, index=False, encoding='utf-8')
            kodiert = df['label'].isin(['HATE', 'NON_HATE']).sum()
            hate_count = (df['label'] == 'HATE').sum()
            print(f"\n  Gespeichert. Fortschritt: {kodiert}/{len(df)} kodiert  "
                  f"({hate_count} HATE, {kodiert - hate_count} NON_HATE)")
            sys.exit(0)
        else:
            print("  Ungültige Eingabe. Bitte h, n, s oder q drücken.")

    # Alle 10 Texte automatisch speichern
    if (i + 1) % 10 == 0:
        df.to_csv(INPUT_FILE, index=False, encoding='utf-8')
        kodiert = df['label'].isin(['HATE', 'NON_HATE']).sum()
        hate_count = (df['label'] == 'HATE').sum()
        print(f"\n  [Auto-Speicherung bei Text {i+1}]  "
              f"{kodiert} kodiert, davon {hate_count} HATE")
        input("  Enter zum Weitermachen...")

df.to_csv(INPUT_FILE, index=False, encoding='utf-8')
kodiert = df['label'].isin(['HATE', 'NON_HATE']).sum()
hate_count = (df['label'] == 'HATE').sum()
print(f"\n  Fertig! {kodiert}/{len(df)} Texte kodiert "
      f"({hate_count} HATE, {kodiert - hate_count} NON_HATE).")
print(f"  Gespeichert in: {INPUT_FILE}\n")
print("  Nächster Schritt (nach vollständiger Annotation):")
print("    python src/retrain.py")
