import pandas as pd
import os

INPUT_FILE = os.path.join(os.path.dirname(__file__), 'intra_rater_kodierung.csv')

df = pd.read_csv(INPUT_FILE)

# Fortschritt: überspringe bereits kodierte Zeilen
bereits_kodiert = df['label_runde2'].notna() & (df['label_runde2'] != '')
start_idx = bereits_kodiert.sum()

print("\n" + "="*60)
print("  INTRA-RATER KODIERUNG")
print("="*60)
print(f"  Gesamt: {len(df)} Texte | Bereits kodiert: {start_idx}")
print("\n  Tasten:")
print("    h  -> HATE")
print("    n  -> NON_HATE")
print("    s  -> Überspringen (später)")
print("    q  -> Speichern & Beenden")
print("="*60)
input("\n  Enter drücken zum Starten...")

for i in range(start_idx, len(df)):
    row = df.iloc[i]

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n[{i+1}/{len(df)}]  (bereits kodiert: {i})\n")
    print("-"*60)
    # Zeige nur die ersten 800 Zeichen des Textes
    text = str(row['text'])
    if len(text) > 800:
        print(text[:800])
        print(f"\n... [Text gekürzt, {len(text)} Zeichen gesamt]")
    else:
        print(text)
    print("-"*60)
    print("\n  [h] HATE    [n] NON_HATE    [s] Überspringen    [q] Beenden\n")

    while True:
        taste = input("  Eingabe: ").strip().lower()
        if taste == 'h':
            df.at[i, 'label_runde2'] = 'HATE'
            break
        elif taste == 'n':
            df.at[i, 'label_runde2'] = 'NON_HATE'
            break
        elif taste == 's':
            break
        elif taste == 'q':
            df.to_csv(INPUT_FILE, index=False)
            print(f"\n  Gespeichert. Fortschritt: {i}/{len(df)} Texte kodiert.")
            exit()
        else:
            print("  Ungültige Eingabe. Bitte h, n, s oder q drücken.")

    # Alle 10 Texte automatisch speichern
    if (i + 1) % 10 == 0:
        df.to_csv(INPUT_FILE, index=False)
        print(f"\n  [Auto-Speicherung bei Text {i+1}]")
        input("  Enter zum Weitermachen...")

df.to_csv(INPUT_FILE, index=False)
kodiert = (df['label_runde2'].notna() & (df['label_runde2'] != '')).sum()
print(f"\n  Fertig! {kodiert}/{len(df)} Texte kodiert.")
print(f"  Gespeichert in: {INPUT_FILE}\n")
