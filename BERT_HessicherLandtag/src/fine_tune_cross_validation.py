import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# Configuration
MODEL_NAME = "Hate-speech-CNERG/dehatebert-mono-german"
OUTPUT_DIR = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\fine_tuned_model_cv"
TRAINING_DATA_PATH = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\labeled_data.parquet"
TRAINING_OUTPUT_FILE = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\training_output_cv.txt"
TRAINING_METRICS_FILE = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\training_metrics_cv.json"
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
N_FOLDS = 5
RANDOM_STATE = 42

# Create output file and write header
def log_output(message, print_to_console=True):
    """Write message to both console and output file"""
    if print_to_console:
        print(message)
    with open(TRAINING_OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(message + '\n')

# Initialize output file
with open(TRAINING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("")

log_output("="*80)
log_output("BERT FINE-TUNING SCRIPT - CROSS-VALIDATION")
log_output("="*80)
log_output(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_output("")

# 1. Check device availability
if torch.cuda.is_available():
    device = 0
    log_output(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    log_output("No GPU available. Using CPU for training.")
    log_output("Warning: Training on CPU will be significantly slower.")

# 2. Load training data
log_output(f"\nLoading training data from: {TRAINING_DATA_PATH}")
try:
    df = pd.read_parquet(TRAINING_DATA_PATH)
    log_output(f"Loaded {len(df)} labeled examples")
    log_output(f"Columns: {df.columns.tolist()}")

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must have 'text' and 'label' columns")

    log_output(f"\nLabel distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        log_output(f"  {label}: {count} ({count/len(df)*100:.2f}%)")

except FileNotFoundError:
    log_output(f"Error: Could not find training data at {TRAINING_DATA_PATH}")
    exit(1)

# 3. Prepare label mapping
unique_labels = df['label'].unique()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_labels)

log_output(f"\nLabel mapping:")
for label, idx in label2id.items():
    log_output(f"  {label} -> {idx}")

df['label_id'] = df['label'].map(label2id)

# 4. Load tokenizer
log_output(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 5. Tokenize function
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

# 6. Define metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Custom callback to log training progress
class OutputLoggingCallback(TrainerCallback):
    def __init__(self):
        self.training_logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                **logs
            }
            self.training_logs.append(log_entry)

    def on_epoch_end(self, args, state, control, **kwargs):
        log_output(f"  Epoch {state.epoch:.0f} completed.")

# 7. Cross-Validation
log_output(f"\n{'='*80}")
log_output(f"STARTING {N_FOLDS}-FOLD CROSS-VALIDATION")
log_output(f"{'='*80}")
log_output(f"\nConfiguration:")
log_output(f"  Model: {MODEL_NAME}")
log_output(f"  Epochs per fold: {NUM_EPOCHS}")
log_output(f"  Batch size: {BATCH_SIZE}")
log_output(f"  Learning rate: {LEARNING_RATE}")
log_output(f"  Max sequence length: {MAX_LENGTH}")
log_output(f"  Number of folds: {N_FOLDS}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

all_results = []
all_predictions = []
all_true_labels = []
fold_metrics = []

training_start_time = datetime.now()

for fold, (train_idx, val_idx) in enumerate(skf.split(df['text'], df['label_id'])):
    fold_start_time = datetime.now()
    log_output(f"\n{'='*40}")
    log_output(f"FOLD {fold + 1}/{N_FOLDS}")
    log_output(f"{'='*40}")

    # Split data
    train_texts = df['text'].iloc[train_idx].tolist()
    val_texts = df['text'].iloc[val_idx].tolist()
    train_labels = df['label_id'].iloc[train_idx].tolist()
    val_labels = df['label_id'].iloc[val_idx].tolist()

    log_output(f"  Training samples: {len(train_texts)}")
    log_output(f"  Validation samples: {len(val_texts)}")

    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    val_dataset = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels
    })

    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Load fresh model for each fold
    log_output(f"  Loading fresh model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    fold_output_dir = f"{OUTPUT_DIR}/fold_{fold + 1}"
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=f'{fold_output_dir}/logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none",
    )

    # Initialize trainer
    logging_callback = OutputLoggingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[logging_callback]
    )

    # Train
    log_output(f"  Training...")
    trainer.train()

    # Evaluate
    eval_results = trainer.evaluate()
    log_output(f"  Results: Accuracy={eval_results['eval_accuracy']:.4f}, F1={eval_results['eval_f1']:.4f}")

    # Get predictions
    predictions_output = trainer.predict(val_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=1)

    all_predictions.extend(predictions.tolist())
    all_true_labels.extend(val_labels)

    fold_end_time = datetime.now()
    fold_duration = fold_end_time - fold_start_time

    fold_metrics.append({
        'fold': fold + 1,
        'accuracy': eval_results['eval_accuracy'],
        'f1': eval_results['eval_f1'],
        'precision': eval_results['eval_precision'],
        'recall': eval_results['eval_recall'],
        'duration': str(fold_duration)
    })

    all_results.append(eval_results)

    # Save best model from this fold if it's the best so far
    if fold == 0 or eval_results['eval_f1'] > max(r['eval_f1'] for r in all_results[:-1]):
        log_output(f"  Saving best model (F1={eval_results['eval_f1']:.4f})...")
        trainer.save_model(f"{OUTPUT_DIR}/best_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")

training_end_time = datetime.now()
training_duration = training_end_time - training_start_time

# 8. Calculate overall metrics
log_output(f"\n{'='*80}")
log_output("CROSS-VALIDATION RESULTS")
log_output(f"{'='*80}")

# Per-fold results
log_output("\nPer-Fold Results:")
log_output(f"{'Fold':<6} {'Accuracy':<12} {'F1':<12} {'Precision':<12} {'Recall':<12}")
log_output("-" * 54)
for fm in fold_metrics:
    log_output(f"{fm['fold']:<6} {fm['accuracy']:<12.4f} {fm['f1']:<12.4f} {fm['precision']:<12.4f} {fm['recall']:<12.4f}")

# Average metrics
avg_accuracy = np.mean([r['eval_accuracy'] for r in all_results])
avg_f1 = np.mean([r['eval_f1'] for r in all_results])
avg_precision = np.mean([r['eval_precision'] for r in all_results])
avg_recall = np.mean([r['eval_recall'] for r in all_results])

std_accuracy = np.std([r['eval_accuracy'] for r in all_results])
std_f1 = np.std([r['eval_f1'] for r in all_results])
std_precision = np.std([r['eval_precision'] for r in all_results])
std_recall = np.std([r['eval_recall'] for r in all_results])

log_output("-" * 54)
log_output(f"{'Mean':<6} {avg_accuracy:<12.4f} {avg_f1:<12.4f} {avg_precision:<12.4f} {avg_recall:<12.4f}")
log_output(f"{'Std':<6} {std_accuracy:<12.4f} {std_f1:<12.4f} {std_precision:<12.4f} {std_recall:<12.4f}")

# Overall classification report
log_output(f"\n{'='*80}")
log_output("OVERALL CLASSIFICATION REPORT (All Folds Combined)")
log_output(f"{'='*80}\n")

target_names = [id2label[i] for i in range(num_labels)]
class_report = classification_report(all_true_labels, all_predictions, target_names=target_names)
log_output(class_report)

# Confusion matrix
log_output("\nOverall Confusion Matrix:")
cm = confusion_matrix(all_true_labels, all_predictions)
log_output(f"\n{cm}")

# 9. Save metrics to JSON
metrics_summary = {
    'model_name': MODEL_NAME,
    'training_started': training_start_time.strftime('%Y-%m-%d %H:%M:%S'),
    'training_completed': training_end_time.strftime('%Y-%m-%d %H:%M:%S'),
    'training_duration': str(training_duration),
    'configuration': {
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'max_length': MAX_LENGTH,
        'n_folds': N_FOLDS,
        'num_labels': num_labels
    },
    'dataset_info': {
        'total_samples': len(df),
        'label_distribution': {str(k): int(v) for k, v in label_counts.items()}
    },
    'label_mapping': label2id,
    'fold_metrics': fold_metrics,
    'average_metrics': {
        'accuracy': {'mean': avg_accuracy, 'std': std_accuracy},
        'f1': {'mean': avg_f1, 'std': std_f1},
        'precision': {'mean': avg_precision, 'std': std_precision},
        'recall': {'mean': avg_recall, 'std': std_recall}
    },
    'confusion_matrix': cm.tolist(),
    'output_directory': OUTPUT_DIR
}

with open(TRAINING_METRICS_FILE, 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

log_output(f"\nTraining metrics saved to: {TRAINING_METRICS_FILE}")

# 10. Final summary
log_output(f"\n{'='*80}")
log_output("CROSS-VALIDATION COMPLETE!")
log_output(f"{'='*80}")
log_output(f"\nTotal training duration: {training_duration}")
log_output(f"\nFinal Average Metrics:")
log_output(f"  Accuracy: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
log_output(f"  F1-Score: {avg_f1:.4f} (+/- {std_f1:.4f})")
log_output(f"  Precision: {avg_precision:.4f} (+/- {std_precision:.4f})")
log_output(f"  Recall: {avg_recall:.4f} (+/- {std_recall:.4f})")
log_output(f"\nOutput files:")
log_output(f"  - Best model: {OUTPUT_DIR}/best_model")
log_output(f"  - Training log: {TRAINING_OUTPUT_FILE}")
log_output(f"  - Metrics JSON: {TRAINING_METRICS_FILE}")

log_output(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_output("="*80)
