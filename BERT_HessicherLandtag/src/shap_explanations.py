"""
SHAP-Erklärungen für das BERT Hate-Speech-Modell
=================================================
Erklärt einzelne Vorhersagen mit SHAP (SHapley Additive exPlanations):
Welche Token haben die Klassifikation am stärksten beeinflusst?

Standard-Eingabe: Data/evaluation/model_confirmed_hate_migration.csv
  (49 vom Modell bestätigte HATE-Redebeiträge, sortiert nach p_hate)

Ausgabe:
  - HTML-Dateien mit Token-Hervorhebung     →  Data/evaluation/shap/
  - PNG-Zusammenfassung (Bar-Plot)          →  Data/evaluation/shap/
  - CSV mit Token-Gewichten                 →  Data/evaluation/shap/shap_weights.csv

Verwendung:
  python src/shap_explanations.py                        # Top-10 HATE-Reden
  python src/shap_explanations.py --n 20                 # Top-20
  python src/shap_explanations.py --csv eigene.csv --n 5
  python src/shap_explanations.py --filter HATE --n 20
"""

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Pfade & Konfiguration
# ---------------------------------------------------------------------------
BASE_DIR      = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
MODEL_DIR     = BASE_DIR / "fine_tuned_model_retrain" / "best_model"
DEFAULT_CSV   = BASE_DIR / "Data" / "evaluation" / "model_confirmed_hate_migration.csv"
OUT_DIR       = BASE_DIR / "Data" / "evaluation" / "shap"

TARGET_LABEL  = "HATE"

# ---------------------------------------------------------------------------
# Modell + Explainer aufbauen
# ---------------------------------------------------------------------------
def build_explainer(model_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lade Modell von: {model_dir}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    id2label    = model.config.id2label
    label_to_idx = {v: k for k, v in id2label.items()}
    col_nonhate  = label_to_idx.get("NON_HATE", 0)
    col_hate     = label_to_idx.get("HATE", 1)

    def predict_fn(texts: list) -> np.ndarray:
        inputs = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        return np.column_stack([probs[:, col_nonhate], probs[:, col_hate]])

    # shap.explainers.Partition direkt verwenden (umgeht die Python-3.14-
    # inkompatible is_transformers_lm-Prüfung in shap.Explainer)
    masker    = shap.maskers.Text(tokenizer=r"\W+")
    explainer = shap.explainers.Partition(predict_fn, masker=masker,
                                          output_names=["NON_HATE", "HATE"])
    print("SHAP Partition-Explainer initialisiert.")
    return predict_fn, explainer


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
_TOC_RE = re.compile(r'[A-ZÄÖÜ][a-zäöüß]+(?:\s+\S+){0,4}\.{5,}\d', re.MULTILINE)

def is_toc_page(text: str) -> bool:
    return len(_TOC_RE.findall(str(text))) >= 3


def load_texts(csv_path: Path, n: int, label_filter: str | None,
               no_toc_filter: bool) -> pd.DataFrame:
    df = pd.read_parquet(str(csv_path)) if str(csv_path).endswith(".parquet") \
         else pd.read_csv(csv_path)

    text_col  = next((c for c in ["text", "context"] if c in df.columns), None)
    label_col = next((c for c in ["label", "hate_label"] if c in df.columns), None)

    if not text_col:
        raise ValueError(f"Keine 'text'- oder 'context'-Spalte. Gefunden: {df.columns.tolist()}")

    df = df[df[text_col].notna()]
    df = df[df[text_col].str.split().str.len() >= 20]

    if label_filter and label_col:
        df = df[df[label_col] == label_filter]
        print(f"Gefiltert auf '{label_filter}': {len(df)} Texte")

    if not no_toc_filter:
        before = len(df)
        df = df[~df[text_col].apply(is_toc_page)]
        if before - len(df):
            print(f"TOC-Filter: {before - len(df)} Seiten entfernt, {len(df)} verbleiben")

    # Nach p_hate absteigend sortieren, falls vorhanden
    if "p_hate" in df.columns:
        df = df.sort_values("p_hate", ascending=False)

    return df.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Erklärungen berechnen & speichern
# ---------------------------------------------------------------------------
def explain_texts(df: pd.DataFrame, text_col: str, predict_fn, explainer, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = df[text_col].tolist()

    print(f"\nBerechne SHAP-Werte für {len(texts)} Redebeiträge ...")
    shap_values = explainer(texts)

    all_rows = []

    for i, text in enumerate(texts):
        probs_row  = predict_fn([text])[0]
        p_nonhate, p_hate = float(probs_row[0]), float(probs_row[1])
        pred_label = "HATE" if p_hate >= 0.5 else "NON_HATE"

        # Metadaten aus der Quell-Zeile
        meta = df.iloc[i]
        speaker = meta.get("speaker", "") if hasattr(meta, "get") else ""
        party   = meta.get("party",   "") if hasattr(meta, "get") else ""
        p_hate_orig = meta.get("p_hate", p_hate) if hasattr(meta, "get") else p_hate

        print(f"\n[{i+1}/{len(texts)}] {speaker} ({party})" if speaker
              else f"\n[{i+1}/{len(texts)}]")
        print(f"  '{text[:80]}{'...' if len(text) > 80 else ''}'")
        print(f"  Vorhersage: {pred_label}  (P(HATE)={p_hate:.4f})")

        sv     = shap_values[i, :, TARGET_LABEL]
        tokens = sv.data
        weights = sv.values

        top = sorted(zip(tokens, weights), key=lambda x: abs(x[1]), reverse=True)[:10]
        print("  Top-Token (|SHAP|):")
        for tok, w in top:
            print(f"    {'+'if w>0 else '-'}{abs(w):.4f}  '{tok}'")

        # HTML speichern
        html_path = out_dir / f"shap_text_{i+1:03d}.html"
        html_content = shap.plots.text(sv, display=False)
        if html_content:
            html_path.write_text(html_content, encoding="utf-8")
            print(f"  HTML gespeichert: {html_path.name}")

        for tok, w in zip(tokens, weights):
            all_rows.append({
                "text_idx":    i + 1,
                "speaker":     speaker,
                "party":       party,
                "p_hate_orig": round(float(p_hate_orig), 4),
                "pred_label":  pred_label,
                "p_hate":      round(p_hate, 4),
                "text_preview": text[:120],
                "token":       tok,
                "shap_value":  round(float(w), 6),
            })

    if all_rows:
        df_out = pd.DataFrame(all_rows)
        csv_path = out_dir / "shap_weights.csv"
        df_out.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\nAlle SHAP-Gewichte gespeichert: {csv_path}")

        top_hate = (
            df_out[df_out["shap_value"] > 0]
            .groupby("token")["shap_value"]
            .mean()
            .sort_values(ascending=False)
            .head(20)
        )
        print("\nTop-Token mit stärkstem HATE-Einfluss (Ø SHAP über alle Reden):")
        print(top_hate.to_string())

        _save_bar_plot(top_hate, out_dir)

    return shap_values


def _save_bar_plot(top_series: pd.Series, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_series.index[::-1], top_series.values[::-1], color="#d73027")
    ax.set_xlabel("Ø SHAP-Wert (HATE-Klasse)")
    ax.set_title("Top-20 Token nach HATE-Einfluss\n(Ø über alle erklärten Redebeiträge)",
                 fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    out = out_dir / "shap_top_tokens_bar.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Bar-Plot gespeichert: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SHAP-Erklärungen für BERT Hate-Speech-Modell")
    parser.add_argument("--text",      type=str, default=None,
                        help="Einzelner Text direkt als Argument (überschreibt --csv)")
    parser.add_argument("--csv",       type=str, default=str(DEFAULT_CSV),
                        help=f"Pfad zur CSV oder Parquet (Standard: {DEFAULT_CSV.name})")
    parser.add_argument("--n",         type=int, default=10,
                        help="Anzahl zu erklärender Texte aus CSV (Standard: 10)")
    parser.add_argument("--model",     type=str, default=str(MODEL_DIR),
                        help="Pfad zum Modellverzeichnis")
    parser.add_argument("--filter",    type=str, default=None,
                        choices=["HATE", "NON_HATE"],
                        help="Nur Texte mit diesem Label laden (wenn label-Spalte vorhanden)")
    parser.add_argument("--no-filter", action="store_true",
                        help="TOC-Filter deaktivieren")
    return parser.parse_known_args()[0]


def main():
    args = parse_args()

    predict_fn, explainer = build_explainer(Path(args.model))

    if args.text:
        # Einzelner Text direkt per CLI
        print(f"\nEinzeltext: '{args.text[:80]}...'")
        df = pd.DataFrame([{"text": args.text}])
        text_col = "text"
    else:
        csv_path = Path(args.csv)
        print(f"\nLade Texte aus: {csv_path.name}")
        df = load_texts(csv_path, args.n, args.filter, args.no_filter)
        text_col = next((c for c in ["text", "context"] if c in df.columns))

        if "speaker" in df.columns:
            print("\nRedner in diesem Lauf:")
            for _, row in df.iterrows():
                print(f"  {row['speaker']} ({row.get('party','?')})  p_hate={row.get('p_hate','?')}")

    print(f"\nAusgabe: {OUT_DIR}")
    print("=" * 70)

    explain_texts(df, text_col, predict_fn, explainer, OUT_DIR)

    print("\n" + "=" * 70)
    print("Fertig! HTML-Dateien im Browser öffnen für interaktive Token-Ansicht.")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
