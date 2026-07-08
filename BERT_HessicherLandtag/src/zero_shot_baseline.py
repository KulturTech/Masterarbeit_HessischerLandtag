"""
Baseline-Vergleich: dehatebert-mono-german vs. Fine-Tuned-Modell
=================================================================
Klassifiziert Redebeiträge mit BEIDEN Modellen und vergleicht die Ergebnisse.

Baseline : Hate-speech-CNERG/dehatebert-mono-german  (kein Fine-Tuning)
Fine-Tuned: fine_tuned_model_retrain/best_model       (auf Landtag-Daten trainiert)

Ausgabe:
  Data/evaluation/baseline_comparison.csv    – Vorhersagen beider Modelle pro Text
  Data/evaluation/baseline_metrics.json      – Metriken (wenn Gold-Labels vorhanden)

Verwendung:
  python src/zero_shot_baseline.py                    # manuell annotierte Daten (Gold)
  python src/zero_shot_baseline.py --csv Data/evaluation/afd_migration_speeches.csv
  python src/zero_shot_baseline.py --n 50             # Schnelltest
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR   = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
DEFAULT_INPUT = BASE_DIR / "Data" / "prep_v1" / "speech_segments.parquet"
OUT_CSV    = BASE_DIR / "Data" / "evaluation" / "baseline_comparison.csv"
OUT_JSON   = BASE_DIR / "Data" / "evaluation" / "baseline_metrics.json"

BASELINE_MODEL  = "Hate-speech-CNERG/dehatebert-mono-german"
FINETUNED_MODEL = BASE_DIR / "fine_tuned_model_retrain" / "best_model"

BATCH_SIZE = 16


def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()
    return tokenizer, model, device


def predict(texts, tokenizer, model, device):
    labels, scores = [], []
    id2label = model.config.id2label
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
        for p in probs:
            idx = int(np.argmax(p))
            labels.append(id2label[idx])
            scores.append(round(float(p[idx]), 4))
        if (i // BATCH_SIZE + 1) % 20 == 0:
            print(f"  {i + len(batch)}/{len(texts)}")
    return labels, scores


def metrics_report(y_true, y_pred, model_name):
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n--- {model_name} ---")
    print(classification_report(y_true, y_pred, labels=['NON_HATE', 'HATE']))
    cm = confusion_matrix(y_true, y_pred, labels=['NON_HATE', 'HATE'])
    print(f"Konfusionsmatrix [[TN FP] [FN TP]]: {cm.tolist()}")
    return classification_report(y_true, y_pred, labels=['NON_HATE', 'HATE'],
                                 output_dict=True), cm.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=None)
    parser.add_argument('--n',   type=int, default=0)
    args, _ = parser.parse_known_args()

    print('=' * 65)
    print('Baseline-Vergleich: dehatebert vs. Fine-Tuned')
    print('=' * 65)

    # Daten laden
    if args.csv:
        src = args.csv
        df = pd.read_csv(src) if src.endswith('.csv') else pd.read_parquet(src)
    else:
        df = pd.read_parquet(DEFAULT_INPUT)
    df = df[df['text'].notna()].copy()
    if args.n:
        df = df.head(args.n)
    print(f"Texte: {len(df)}")

    label_col = next((c for c in ['label', 'hate_label'] if c in df.columns), None)
    if label_col:
        print(f"Gold-Labels gefunden (Spalte '{label_col}'): "
              f"{df[label_col].value_counts().to_dict()}")

    texts = df['text'].tolist()

    # ── Baseline: dehatebert (kein Fine-Tuning) ──────────────────────────
    print(f"\n[1/2] Lade Baseline-Modell: {BASELINE_MODEL}")
    tok_b, mod_b, dev_b = load_model(BASELINE_MODEL)
    print(f"Device: {dev_b}  |  Labels: {mod_b.config.id2label}")
    print("Klassifiziere...")
    df['base_label'], df['base_score'] = predict(texts, tok_b, mod_b, dev_b)
    del mod_b  # Speicher freigeben

    print(f"\nBaseline-Verteilung:\n{df['base_label'].value_counts().to_string()}")

    # ── Fine-Tuned ────────────────────────────────────────────────────────
    print(f"\n[2/2] Lade Fine-Tuned-Modell: {FINETUNED_MODEL}")
    tok_f, mod_f, dev_f = load_model(FINETUNED_MODEL)
    print(f"Device: {dev_f}  |  Labels: {mod_f.config.id2label}")
    print("Klassifiziere...")
    df['ft_label'], df['ft_score'] = predict(texts, tok_f, mod_f, dev_f)
    del mod_f

    print(f"\nFine-Tuned-Verteilung:\n{df['ft_label'].value_counts().to_string()}")

    # ── Übereinstimmung ───────────────────────────────────────────────────
    agreement = (df['base_label'] == df['ft_label']).mean()
    print(f"\nÜbereinstimmung beider Modelle: {agreement:.1%}")

    # ── Metriken gegen Gold-Labels ────────────────────────────────────────
    metrics = {}
    if label_col:
        valid = df[df[label_col].isin(['HATE', 'NON_HATE'])].copy()
        print(f"\nEvaluation gegen Gold-Labels ({len(valid)} Texte):")
        rep_b, cm_b = metrics_report(valid[label_col], valid['base_label'], 'Baseline (dehatebert)')
        rep_f, cm_f = metrics_report(valid[label_col], valid['ft_label'],   'Fine-Tuned')

        metrics = {
            'baseline':   {'report': rep_b, 'confusion_matrix': cm_b},
            'fine_tuned': {'report': rep_f, 'confusion_matrix': cm_f},
            'agreement':  round(agreement, 4),
        }
        OUT_JSON.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nMetriken gespeichert: {OUT_JSON}")

    df.to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(f"Ergebnisse gespeichert: {OUT_CSV}")


if __name__ == '__main__':
    main()
