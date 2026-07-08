"""
Automatische Annotation mit dehatebert-mono-german
===================================================
Labeliert alle noch nicht annotierten Texte in annotation_sample.csv
mit dem vortrainierten dehatebert-Modell.

  P(HATE) >= threshold  --> HATE
  P(HATE) <  threshold  --> NON_HATE

Ausführung:
  python src/auto_annotate.py                  # Standard-Schwelle 0.30
  python src/auto_annotate.py --threshold 0.50 # Konservativere Schwelle
  python src/auto_annotate.py --dry-run        # Nur anzeigen, nicht speichern
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_DIR       = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
ANNOTATION_CSV = BASE_DIR / "Data" / "annotation" / "annotation_sample.csv"
MODEL_NAME     = "Hate-speech-CNERG/dehatebert-mono-german"
BATCH_SIZE     = 16
MAX_LENGTH     = 256


def score_texts(texts, tokenizer, model, device):
    hate_idx = next(k for k, v in model.config.id2label.items() if v == "HATE")
    probs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Scoring", unit="batch"):
        batch = texts[i: i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        probs.append(p[:, hate_idx])
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="P(HATE)-Schwelle fuer HATE-Label (Standard: 0.30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur anzeigen, nicht in CSV speichern")
    args = parser.parse_args()

    df = pd.read_csv(ANNOTATION_CSV)
    unlabeled = df["label"].isna() | (df["label"] == "")
    todo = df[unlabeled].copy()

    print(f"annotation_sample.csv: {len(df)} Texte gesamt")
    print(f"  Bereits annotiert:  {(~unlabeled).sum()}")
    print(f"  Noch zu annotieren: {len(todo)}")

    if len(todo) == 0:
        print("Nichts zu tun — alle Texte bereits annotiert.")
        return

    # Scores bereits im CSV vorhanden?
    if "p_hate" in todo.columns and todo["p_hate"].notna().all():
        print(f"\nVerwende gecachte P(HATE)-Scores aus CSV...")
        scores = todo["p_hate"].values
    else:
        print(f"\nLade Modell: {MODEL_NAME}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        scores = score_texts(todo["text"].tolist(), tokenizer, model, device)

    labels = np.where(scores >= args.threshold, "HATE", "NON_HATE")
    n_hate    = (labels == "HATE").sum()
    n_nonhate = (labels == "NON_HATE").sum()

    print(f"\nSchwelle: P(HATE) >= {args.threshold}")
    print(f"  -> HATE:     {n_hate}")
    print(f"  -> NON_HATE: {n_nonhate}")

    print(f"\nP(HATE)-Verteilung der neuen Texte:")
    print(f"  > 0.50: {(scores > 0.50).sum()}")
    print(f"  > 0.30: {(scores > 0.30).sum()}")
    print(f"  > 0.15: {(scores > 0.15).sum()}")
    print(f"  <= 0.15: {(scores <= 0.15).sum()}")

    if args.dry_run:
        print("\n[DRY RUN] Keine Änderungen gespeichert.")
        if n_hate > 0:
            print("\nBeispiel HATE-Texte (Top 5 nach P(HATE)):")
            top_idx = np.argsort(scores)[::-1][:5]
            for idx in top_idx:
                if scores[idx] >= args.threshold:
                    print(f"  P={scores[idx]:.3f}: {todo.iloc[idx]['text'][:120]}...")
        return

    df.loc[unlabeled, "label"] = labels
    df.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8")

    total_hate    = (df["label"] == "HATE").sum()
    total_nonhate = (df["label"] == "NON_HATE").sum()
    print(f"\n[OK] Gespeichert: {ANNOTATION_CSV}")
    print(f"     Gesamt annotiert: {len(df)}")
    print(f"     HATE:     {total_hate}")
    print(f"     NON_HATE: {total_nonhate}")
    print(f"\nNaechster Schritt: python src/retrain.py")


if __name__ == "__main__":
    main()
