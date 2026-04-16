"""
Neutraining auf manuell annotierten Daten
==========================================
Liest die fertig annotierten Texte aus Data/annotation/annotation_sample.csv
und trainiert ein neues Modell mit 5-Fold Cross-Validation.

Voraussetzung:
  - Data/annotation/annotation_sample.csv ist vollständig annotiert
    (Spalte 'label' enthält 'HATE' oder 'NON_HATE' für alle gewünschten Texte)
  - Mindestens 20 HATE-Beispiele vorhanden (empfohlen: ≥ 40)

Ausführung:
  python src/retrain.py
  python src/retrain.py --epochs 5 --batch 16  # Anpassungen
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix,
                              precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainerCallback, TrainingArguments)

# ---------------------------------------------------------------------------
# Pfade & Defaults
# ---------------------------------------------------------------------------
BASE_DIR       = Path(r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag")
ANNOTATION_CSV = BASE_DIR / "Data" / "annotation" / "annotation_sample.csv"
OUTPUT_DIR     = BASE_DIR / "fine_tuned_model_retrain"
METRICS_FILE   = BASE_DIR / "Data" / "annotation" / "retrain_metrics.json"
LOG_FILE       = BASE_DIR / "Data" / "annotation" / "retrain_log.txt"

BASE_MODEL  = "Hate-speech-CNERG/dehatebert-mono-german"
MAX_LENGTH  = 256
LABEL2ID    = {'NON_HATE': 0, 'HATE': 1}
ID2LABEL    = {0: 'NON_HATE', 1: 'HATE'}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LOG_FILE, 'w', encoding='utf-8') as _f:
    _f.write('')

def log(msg: str, console: bool = True):
    if console:
        print(msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    # Zusätzlich HATE-spezifische Metriken
    p_hate, r_hate, f1_hate, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=[1])
    return {
        'accuracy':       accuracy_score(labels, preds),
        'f1':             f1,
        'precision':      p,
        'recall':         r,
        'f1_hate':        float(f1_hate[0]),
        'precision_hate': float(p_hate[0]),
        'recall_hate':    float(r_hate[0]),
    }


class LogCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        log(f"    Epoch {state.epoch:.0f} abgeschlossen.", console=False)


# ---------------------------------------------------------------------------
# Trainer mit Class Weights
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """Trainer mit gewichteter Cross-Entropy — gleicht Klassenimbalanz aus."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',  type=int,   default=3)
    parser.add_argument('--batch',   type=int,   default=8)
    parser.add_argument('--lr',      type=float, default=2e-5)
    parser.add_argument('--folds',   type=int,   default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = datetime.now()
    log("=" * 70)
    log("NEUTRAINING AUF MANUELL ANNOTIERTEN DATEN")
    log("=" * 70)
    log(f"Start: {start:%Y-%m-%d %H:%M:%S}")

    # 1. Annotierte Daten laden
    log(f"\nLade annotierte Daten: {ANNOTATION_CSV}")
    df_all = pd.read_csv(ANNOTATION_CSV)
    df = df_all[df_all['label'].isin(['HATE', 'NON_HATE'])].copy()
    log(f"  Annotierte Texte gesamt: {len(df)}")
    counts = df['label'].value_counts()
    for lbl, cnt in counts.items():
        log(f"    {lbl}: {cnt}")

    if len(df) < 30:
        log("\nFEHLER: Zu wenige annotierte Texte (< 30). Bitte mehr annotieren.")
        return
    if counts.get('HATE', 0) < 10:
        log(f"\nWARNUNG: Nur {counts.get('HATE', 0)} HATE-Beispiele. "
            f"Empfehlung: ≥ 20 für sinnvolles Training.")

    df['label_id'] = df['label'].map(LABEL2ID)
    texts  = df['text'].tolist()
    labels = df['label_id'].tolist()

    # Class Weights berechnen
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=np.array(labels))
    class_weights = torch.tensor(cw, dtype=torch.float)
    log(f"\nClass Weights: NON_HATE={cw[0]:.3f}, HATE={cw[1]:.3f}")

    # 2. Tokenizer laden
    log(f"\nLade Tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(examples):
        return tokenizer(examples['text'], padding='max_length',
                         truncation=True, max_length=MAX_LENGTH)

    # 3. Cross-Validation
    n_folds = min(args.folds, counts.min())   # kann nicht mehr Folds als kleinste Klasse haben
    log(f"\n{n_folds}-Fold Cross-Validation")
    log(f"  Epochen: {args.epochs}  |  Batch: {args.batch}  |  LR: {args.lr}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds, all_labels = [], []
    fold_metrics = []
    best_f1 = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
        log(f"\n{'-'*40}")
        log(f"Fold {fold+1}/{n_folds}  "
            f"(train={len(train_idx)}, val={len(val_idx)})")

        train_ds = Dataset.from_dict({
            'text':  [texts[i] for i in train_idx],
            'label': [labels[i] for i in train_idx],
        }).map(tokenize, batched=True)
        val_ds = Dataset.from_dict({
            'text':  [texts[i] for i in val_idx],
            'label': [labels[i] for i in val_idx],
        }).map(tokenize, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, num_labels=2,
            id2label=ID2LABEL, label2id=LABEL2ID
        )

        fold_dir = str(OUTPUT_DIR / f"fold_{fold+1}")
        training_args = TrainingArguments(
            output_dir=fold_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            save_total_limit=1,
            report_to='none',
            logging_steps=20,
        )

        trainer = WeightedTrainer(
            model=model, args=training_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[LogCallback()],
            class_weights=class_weights,
        )
        trainer.train()

        res = trainer.evaluate()
        f1  = res['eval_f1']
        log(f"  Accuracy={res['eval_accuracy']:.4f}  F1={f1:.4f}  "
            f"Precision={res['eval_precision']:.4f}  Recall={res['eval_recall']:.4f}  "
            f"| HATE → P={res['eval_precision_hate']:.4f}  R={res['eval_recall_hate']:.4f}  F1={res['eval_f1_hate']:.4f}")

        fold_metrics.append({
            'fold': fold + 1,
            'accuracy': res['eval_accuracy'], 'f1': f1,
            'precision': res['eval_precision'], 'recall': res['eval_recall'],
        })

        # Predictions sammeln
        pred_out = trainer.predict(val_ds)
        fold_preds = np.argmax(pred_out.predictions, axis=1).tolist()
        all_preds.extend(fold_preds)
        all_labels.extend([labels[i] for i in val_idx])

        # Bestes Modell speichern
        if f1 > best_f1:
            best_f1 = f1
            log(f"  → Neues bestes Modell (F1={f1:.4f}), speichere...")
            trainer.save_model(str(OUTPUT_DIR / "best_model"))
            tokenizer.save_pretrained(str(OUTPUT_DIR / "best_model"))

    # 4. Gesamtergebnis
    end = datetime.now()
    log(f"\n{'='*70}")
    log("GESAMTERGEBNIS (alle Folds kombiniert)")
    log(f"{'='*70}")
    log(f"\n{classification_report(all_labels, all_preds, target_names=['NON_HATE', 'HATE'])}")
    log(f"\nKonfusionsmatrix:\n{confusion_matrix(all_labels, all_preds)}")

    avg = lambda key: float(np.mean([m[key] for m in fold_metrics]))
    std = lambda key: float(np.std([m[key] for m in fold_metrics]))
    log(f"\nDurchschnittliche Metriken über {n_folds} Folds:")
    for metric in ['accuracy', 'f1', 'precision', 'recall']:
        log(f"  {metric:<12}: {avg(metric):.4f} ± {std(metric):.4f}")

    # 5. Metriken speichern
    metrics_summary = {
        'started': start.strftime('%Y-%m-%d %H:%M:%S'),
        'finished': end.strftime('%Y-%m-%d %H:%M:%S'),
        'base_model': BASE_MODEL,
        'config': vars(args),
        'dataset': {'total': len(df), 'label_dist': counts.to_dict()},
        'fold_metrics': fold_metrics,
        'avg_metrics': {m: {'mean': avg(m), 'std': std(m)}
                        for m in ['accuracy', 'f1', 'precision', 'recall']},
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'best_model_path': str(OUTPUT_DIR / 'best_model'),
    }
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    log(f"\n[OK] Metriken gespeichert: {METRICS_FILE}")
    log(f"[OK] Bestes Modell:         {OUTPUT_DIR / 'best_model'}")
    log(f"[OK] Trainingslog:          {LOG_FILE}")
    log(f"\nGesamtdauer: {end - start}")
    log("=" * 70)


if __name__ == '__main__':
    main()
