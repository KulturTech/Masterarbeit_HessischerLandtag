"""
GermEval 2018 migrationsrelevante Beispiele hinzufügen
=======================================================
Lädt GermEval 2018 von HuggingFace, filtert nach Migrations-Keywords
und fügt passende Beispiele zu annotation_sample.csv hinzu.

  OFFENSE -> HATE
  OTHER   -> NON_HATE

Ausführung:
  python src/add_germeval.py
  python src/add_germeval.py --max-hate 100 --max-nonhate 100
  python src/add_germeval.py --no-filter   # alle GermEval-Texte (kein Keyword-Filter)
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

BASE_DIR       = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
ANNOTATION_CSV = BASE_DIR / "Data" / "annotation" / "annotation_sample.csv"

_MIGRATION_RE = re.compile('|'.join([
    r'\b(immigrant\w*|einwanderer\w*|einwanderung\w*|zuwander\w*)\b',
    r'\b(migrant\w*|migration\w*)\b',
    r'\b(flüchtling\w*|fluechtling\w*|asyl\w*)\b',
    r'\b(ausländer\w*|auslaender\w*|ausländisch\w*)\b',
    r'\b(geflüchtete\w*|schutzsuchende\w*)\b',
    r'\b(integration\w*)\b',
    r'\b(abschiebung\w*|rückführung\w*|rueckführung\w*)\b',
    r'\b(grenzsicherung\w*|grenzschutz\w*)\b',
    r'\b(asylpolitik\w*|migrationspolitik\w*|ausländerpolitik\w*)\b',
    r'\b(überfremdung\w*|ueberfremdung\w*)\b',
    r'\b(islamis\w*|muslim\w*|arabisch\w*)\b',
]), re.IGNORECASE)

GERMEVAL_DATASETS = [
    ("philschmid/germeval18", None),
    ("deepset/germeval_2018_A", None),
    ("gwlms/germeval2018", None),
]


def load_germeval():
    for name, config in GERMEVAL_DATASETS:
        try:
            print(f"  Versuche: {name} ...", end=" ", flush=True)
            ds = load_dataset(name, config, trust_remote_code=True)
            print("OK")
            return ds
        except Exception as e:
            print(f"Fehler ({e})")
    raise RuntimeError("Kein GermEval-Datensatz gefunden. Prüfe Internetverbindung.")


def normalize_label(label_val) -> str:
    """Mappt verschiedene GermEval-Label-Formate auf HATE/NON_HATE."""
    s = str(label_val).upper()
    if any(x in s for x in ["OFFENSE", "1", "TRUE", "HATE", "OFFENS"]):
        return "HATE"
    return "NON_HATE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hate",    type=int, default=150,
                        help="Max. HATE-Beispiele aus GermEval (Standard: 150)")
    parser.add_argument("--max-nonhate", type=int, default=100,
                        help="Max. NON_HATE-Beispiele aus GermEval (Standard: 100)")
    parser.add_argument("--no-filter",   action="store_true",
                        help="Kein Migrations-Keyword-Filter anwenden")
    args = parser.parse_args()

    print("=" * 70)
    print("GERMEVAL 2018 -> annotation_sample.csv")
    print("=" * 70)

    # 1. Bestehendes Sample laden
    existing = pd.read_csv(ANNOTATION_CSV)
    existing_texts = set(existing["text"].str.strip())
    print(f"\nBestehendes Sample: {len(existing)} Texte "
          f"({(existing['label']=='HATE').sum()} HATE, "
          f"{(existing['label']=='NON_HATE').sum()} NON_HATE)")

    # 2. GermEval laden
    print("\nLade GermEval 2018 ...")
    ds = load_germeval()

    # Alle Splits zusammenführen
    splits = list(ds.keys())
    print(f"  Splits gefunden: {splits}")
    frames = []
    for split in splits:
        df_s = ds[split].to_pandas()
        frames.append(df_s)
    df_raw = pd.concat(frames, ignore_index=True)
    print(f"  Gesamt: {len(df_raw)} Einträge")
    print(f"  Spalten: {df_raw.columns.tolist()}")

    # Text-Spalte finden
    text_col = next((c for c in df_raw.columns if c.lower() in ["text", "tweet", "sentence"]), None)
    if text_col is None:
        raise ValueError(f"Keine Text-Spalte gefunden. Spalten: {df_raw.columns.tolist()}")

    # Label-Spalte finden
    label_col = next((c for c in df_raw.columns
                      if c.lower() in ["label", "labels", "coarse", "binary",
                                       "offense", "offensive"]), None)
    if label_col is None:
        raise ValueError(f"Keine Label-Spalte gefunden. Spalten: {df_raw.columns.tolist()}")

    print(f"  Verwende: text='{text_col}', label='{label_col}'")
    print(f"  Label-Werte: {df_raw[label_col].value_counts().to_dict()}")

    # 3. Labels normalisieren
    df_raw["label_norm"] = df_raw[label_col].apply(normalize_label)
    df_raw["text_clean"] = df_raw[text_col].astype(str).str.strip()

    # 4. Duplikate mit bestehendem Sample entfernen
    df_raw = df_raw[~df_raw["text_clean"].isin(existing_texts)].copy()
    print(f"\nNach Duplikat-Filter: {len(df_raw)} neue Texte")

    # 5. Migrations-Keyword-Filter
    if not args.no_filter:
        mask = df_raw["text_clean"].apply(lambda t: bool(_MIGRATION_RE.search(t)))
        df_filtered = df_raw[mask].copy()
        print(f"Nach Migrations-Filter: {len(df_filtered)} Texte")
        print(f"  HATE:     {(df_filtered['label_norm']=='HATE').sum()}")
        print(f"  NON_HATE: {(df_filtered['label_norm']=='NON_HATE').sum()}")
    else:
        df_filtered = df_raw.copy()
        print("Kein Keyword-Filter angewendet.")

    if len(df_filtered) == 0:
        print("\nKeine passenden Texte gefunden.")
        return

    # 6. Auswahl
    hate_pool    = df_filtered[df_filtered["label_norm"] == "HATE"]
    nonhate_pool = df_filtered[df_filtered["label_norm"] == "NON_HATE"]

    n_hate    = min(args.max_hate,    len(hate_pool))
    n_nonhate = min(args.max_nonhate, len(nonhate_pool))

    selected_hate    = hate_pool.sample(n=n_hate,    random_state=42) if n_hate > 0 else hate_pool.iloc[0:0]
    selected_nonhate = nonhate_pool.sample(n=n_nonhate, random_state=42) if n_nonhate > 0 else nonhate_pool.iloc[0:0]
    selected = pd.concat([selected_hate, selected_nonhate]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nAuswahl: {len(selected)} Texte ({n_hate} HATE, {n_nonhate} NON_HATE)")

    # 7. An bestehendes Sample anhängen
    max_id = existing["id"].max() if "id" in existing.columns else 0
    new_rows = pd.DataFrame({
        "id":     range(int(max_id) + 1, int(max_id) + 1 + len(selected)),
        "doc_id": "germeval2018",
        "text":   selected["text_clean"].values,
        "p_hate": None,
        "label":  selected["label_norm"].values,
    })

    combined = pd.concat([existing, new_rows[existing.columns]], ignore_index=True)
    combined.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8")

    total      = len(combined)
    total_hate = (combined["label"] == "HATE").sum()
    total_nonh = (combined["label"] == "NON_HATE").sum()
    print(f"\n[OK] Gespeichert: {ANNOTATION_CSV}")
    print(f"     Gesamt:   {total} Texte")
    print(f"     HATE:     {total_hate}")
    print(f"     NON_HATE: {total_nonh}")
    print(f"\nNaechster Schritt: python src/retrain.py --epochs 3 --folds 5 --batch 16")


if __name__ == "__main__":
    main()
