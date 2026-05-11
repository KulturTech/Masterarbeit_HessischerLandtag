"""
LIME-Erklärungen für das BERT Hate-Speech-Modell
=================================================
Erklärt einzelne Vorhersagen mit LIME (Local Interpretable Model-agnostic
Explanations): Welche Wörter haben die Klassifikation am stärksten beeinflusst?

Ausgabe:
  - HTML-Dateien mit interaktiver Wort-Hervorhebung  →  Data/evaluation/lime/
  - CSV mit Wort-Gewichten für alle erklärten Texte  →  Data/evaluation/lime/lime_weights.csv

Verwendung:
  python src/lime_explanations.py                    # Beispiel-Texte aus dem Skript
  python src/lime_explanations.py --csv Data/evaluation/false_positives_combined_model.csv --n 10
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Pfade & Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR  = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
MODEL_DIR = BASE_DIR / "fine_tuned_model_cv" / "best_model"
OUT_DIR   = BASE_DIR / "Data" / "evaluation" / "lime"

# Klassenreihenfolge: LIME braucht eine feste Reihenfolge
CLASS_NAMES = ["NON_HATE", "HATE"]

# Anzahl der Perturbierungen pro Text (mehr = genauer, aber langsamer)
NUM_SAMPLES = 500

# ---------------------------------------------------------------------------
# Beispiel-Texte (werden verwendet, wenn kein --csv übergeben wird)
# ---------------------------------------------------------------------------
DEFAULT_TEXTS = [
    "Die Ausländer nehmen uns die Arbeitsplätze weg und belasten unser Sozialsystem.",
    "Wir begrüßen die Integration von Flüchtlingen in unsere Gesellschaft.",
    "Die Migrationspolitik der Bundesregierung ist gescheitert und gefährdet die öffentliche Sicherheit.",
    "Asylbewerber sollten faire Chancen auf ein besseres Leben bekommen.",
    "Diese Invasoren kommen nur, um unser Land zu zerstören.",
]


# ---------------------------------------------------------------------------
# TOC-Filter: Inhaltsverzeichnis- und Tagesordnungsseiten ausschließen
# ---------------------------------------------------------------------------
def is_toc_page(text: str) -> bool:
    """Gibt True zurück, wenn der Text eher eine TOC/Agenda-Seite ist als eine Rede."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        return True

    # Zeilen, die nur eine Zahl oder "79." sind (Tagesordnungspunkte)
    numbered_lines = sum(1 for l in lines if re.fullmatch(r'\d+\.?', l))
    if len(lines) > 0 and numbered_lines / len(lines) > 0.15:
        return True

    # Zu kurze Texte (< 40 echte Wörter)
    words = text.split()
    if len(words) < 40:
        return True

    # Zu wenig alphabetische Wörter (≥ 3 Zeichen) — TOC hat viele Zahlen/Kürzel
    real_words = sum(1 for w in words if re.search(r'[a-zA-ZäöüÄÖÜß]{3,}', w))
    if real_words / len(words) < 0.5:
        return True

    return False


# ---------------------------------------------------------------------------
# Modell laden
# ---------------------------------------------------------------------------
def load_model(model_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    print(f"Modell geladen von: {model_dir}")
    print(f"Device: {device}")
    print(f"Labels (id2label): {model.config.id2label}")
    return tokenizer, model, device


# ---------------------------------------------------------------------------
# predict_proba-Wrapper für LIME
# ---------------------------------------------------------------------------
def make_predict_fn(tokenizer, model, device, id2label: dict):
    """
    Gibt eine Funktion zurück, die eine Liste von Texten entgegennimmt und
    ein Array der Form (n, 2) mit [P(NON_HATE), P(HATE)] zurückgibt.
    """
    # Stelle sicher, dass CLASS_NAMES zur Modell-Konfiguration passt
    label_to_idx = {v: k for k, v in id2label.items()}
    col_nonhate = label_to_idx.get("NON_HATE", 0)
    col_hate    = label_to_idx.get("HATE", 1)

    def predict_proba(texts: list) -> np.ndarray:
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        # Spalten in feste Reihenfolge [NON_HATE, HATE] bringen
        return np.column_stack([probs[:, col_nonhate], probs[:, col_hate]])

    return predict_proba


# ---------------------------------------------------------------------------
# LIME erklären
# ---------------------------------------------------------------------------
def explain_texts(texts: list, predict_fn, out_dir: Path, num_samples: int = NUM_SAMPLES):
    """
    Erklärt jede Text-Vorhersage mit LIME und speichert:
      - Eine HTML-Datei pro Text
      - Ein gemeinsames CSV mit allen Wort-Gewichten
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)

    all_weights = []

    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Erkläre Text ({len(text)} Zeichen)...")
        print(f"  '{text[:80]}{'...' if len(text) > 80 else ''}'")

        explanation = explainer.explain_instance(
            text,
            predict_fn,
            num_features=15,       # Top-N Wörter anzeigen
            num_samples=num_samples,
            labels=[1],            # Erklärung für HATE-Klasse
        )

        # Wahrscheinlichkeiten für diesen Text
        probs = predict_fn([text])[0]
        pred_label = CLASS_NAMES[np.argmax(probs)]
        print(f"  Vorhersage: {pred_label}  (P(HATE)={probs[1]:.4f}, P(NON_HATE)={probs[0]:.4f})")

        # HTML speichern
        html_path = out_dir / f"lime_text_{i+1:03d}.html"
        explanation.save_to_file(str(html_path))
        print(f"  HTML gespeichert: {html_path.name}")

        # Wort-Gewichte sammeln
        word_weights = explanation.as_list(label=1)
        for word, weight in word_weights:
            all_weights.append({
                "text_idx":   i + 1,
                "text_preview": text[:100],
                "pred_label": pred_label,
                "p_hate":     round(float(probs[1]), 4),
                "word":       word,
                "weight":     round(float(weight), 6),
            })

    # CSV mit allen Wort-Gewichten
    if all_weights:
        df_weights = pd.DataFrame(all_weights)
        csv_path = out_dir / "lime_weights.csv"
        df_weights.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\nAlle Gewichte gespeichert: {csv_path}")

        # Zusammenfassung: Wörter mit stärkstem HATE-Einfluss über alle Texte
        top_hate_words = (
            df_weights[df_weights["weight"] > 0]
            .groupby("word")["weight"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        print("\nTop-Wörter mit dem stärksten HATE-Einfluss (Durchschnitt über alle Texte):")
        print(top_hate_words.to_string())

    return all_weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LIME-Erklärungen für BERT Hate-Speech-Modell")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Pfad zu einer CSV-Datei mit einer 'text'-Spalte (optional)"
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Anzahl der zu erklärenden Texte aus der CSV (Standard: 5)"
    )
    parser.add_argument(
        "--model", type=str, default=str(MODEL_DIR),
        help="Pfad zum Modellverzeichnis"
    )
    parser.add_argument(
        "--samples", type=int, default=NUM_SAMPLES,
        help=f"LIME-Perturbierungen pro Text (Standard: {NUM_SAMPLES})"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        choices=["HATE", "NON_HATE"],
        help="Nur Texte mit diesem Label aus der CSV laden (wenn 'label'-Spalte vorhanden)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Modell laden
    tokenizer, model, device = load_model(Path(args.model))
    predict_fn = make_predict_fn(tokenizer, model, device, model.config.id2label)

    # Texte bestimmen
    if args.csv:
        df = pd.read_csv(args.csv)
        if "text" not in df.columns:
            raise ValueError(f"CSV hat keine 'text'-Spalte. Gefundene Spalten: {df.columns.tolist()}")
        if args.filter and "label" in df.columns:
            df = df[df["label"] == args.filter]
            print(f"Gefiltert auf '{args.filter}': {len(df)} Texte")
        df = df[df["text"].notna()]
        before = len(df)
        df = df[~df["text"].apply(is_toc_page)]
        print(f"TOC-Filter: {before - len(df)} Seiten entfernt, {len(df)} verbleiben")
        texts = df["text"].head(args.n).tolist()
        print(f"\nLade {len(texts)} Texte aus: {args.csv}")
    else:
        texts = DEFAULT_TEXTS
        print(f"\nVerwende {len(texts)} Standard-Beispieltexte")

    # LIME erklären
    print(f"\nLIME-Konfiguration: {args.samples} Perturbierungen, Top-15 Wörter")
    print(f"Ausgabe: {OUT_DIR}")
    print("=" * 70)

    explain_texts(texts, predict_fn, OUT_DIR, num_samples=args.samples)

    print("\n" + "=" * 70)
    print("Fertig! Öffne die HTML-Dateien im Browser für interaktive Ansicht.")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
