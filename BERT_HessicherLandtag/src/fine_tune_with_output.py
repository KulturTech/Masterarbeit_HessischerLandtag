import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

# Configuration
MODEL_NAME = "Hate-speech-CNERG/dehatebert-mono-german"
OUTPUT_DIR = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\fine_tuned_model"
TRAINING_DATA_PATH = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\labeled_data.parquet"
TRAINING_OUTPUT_FILE = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\training_output.txt"
TRAINING_METRICS_FILE = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\training_metrics.json"
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
TEST_SIZE = 0.2
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
log_output("BERT FINE-TUNING SCRIPT - TRAINING OUTPUT")
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

# 4. Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label_id'].tolist(),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['label_id']
)

log_output(f"\nDataset split:")
log_output(f"  Training samples: {len(train_texts)}")
log_output(f"  Validation samples: {len(val_texts)}")

# 5. Load tokenizer and model
log_output(f"\nLoading model and tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 6. Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

log_output("\nTokenizing datasets...")
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'label': train_labels
})
val_dataset = Dataset.from_dict({
    'text': val_texts,
    'label': val_labels
})

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
log_output("Tokenization complete.")

# 7. Define metrics
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

            # Log to file
            log_message = f"Step {state.global_step}"
            if state.epoch is not None:
                log_message += f" | Epoch {state.epoch:.2f}"
            for key, value in logs.items():
                if isinstance(value, float):
                    log_message += f" | {key}: {value:.4f}"
                else:
                    log_message += f" | {key}: {value}"
            log_output(log_message)

    def on_epoch_end(self, args, state, control, **kwargs):
        log_output(f"\nEpoch {state.epoch} completed.")

# 8. Set up training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=500,
    logging_dir=f'{OUTPUT_DIR}/logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",
)

log_output("\nTraining configuration:")
log_output(f"  Model: {MODEL_NAME}")
log_output(f"  Epochs: {NUM_EPOCHS}")
log_output(f"  Batch size: {BATCH_SIZE}")
log_output(f"  Learning rate: {LEARNING_RATE}")
log_output(f"  Max sequence length: {MAX_LENGTH}")
log_output(f"  Weight decay: 0.01")
log_output(f"  Warmup steps: 500")
log_output(f"  Number of labels: {num_labels}")

# 9. Initialize trainer with callback
logging_callback = OutputLoggingCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[logging_callback]
)

# 10. Train the model
log_output("\n" + "="*80)
log_output("STARTING TRAINING")
log_output("="*80 + "\n")

training_start_time = datetime.now()
trainer.train()
training_end_time = datetime.now()
training_duration = training_end_time - training_start_time

# 11. Evaluate on validation set
log_output("\n" + "="*80)
log_output("FINAL EVALUATION")
log_output("="*80 + "\n")

eval_results = trainer.evaluate()
log_output("Validation Results:")
for key, value in eval_results.items():
    if isinstance(value, float):
        log_output(f"  {key}: {value:.4f}")
    else:
        log_output(f"  {key}: {value}")

# 12. Get predictions for detailed metrics
predictions_output = trainer.predict(val_dataset)
predictions = np.argmax(predictions_output.predictions, axis=1)

# Generate classification report
log_output("\n" + "="*80)
log_output("DETAILED CLASSIFICATION REPORT")
log_output("="*80 + "\n")

target_names = [id2label[i] for i in range(num_labels)]
class_report = classification_report(val_labels, predictions, target_names=target_names)
log_output(class_report)

# Generate confusion matrix
log_output("\nConfusion Matrix:")
cm = confusion_matrix(val_labels, predictions)
log_output(f"\n{cm}")

# Per-class metrics
log_output("\nPer-class metrics:")
for i, label_name in enumerate(target_names):
    mask = np.array(val_labels) == i
    if mask.sum() > 0:
        class_acc = (np.array(predictions)[mask] == i).sum() / mask.sum()
        log_output(f"  {label_name}: {class_acc:.4f} ({mask.sum()} samples)")

# 13. Save the fine-tuned model
log_output(f"\nSaving fine-tuned model to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 14. Save metrics to JSON
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
        'test_size': TEST_SIZE,
        'num_labels': num_labels
    },
    'dataset_info': {
        'total_samples': len(df),
        'training_samples': len(train_texts),
        'validation_samples': len(val_texts),
        'label_distribution': {str(k): int(v) for k, v in label_counts.items()}
    },
    'label_mapping': label2id,
    'final_evaluation': {k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v
                        for k, v in eval_results.items()},
    'training_logs': logging_callback.training_logs,
    'confusion_matrix': cm.tolist(),
    'output_directory': OUTPUT_DIR
}

with open(TRAINING_METRICS_FILE, 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

log_output(f"\nTraining metrics saved to: {TRAINING_METRICS_FILE}")

# 15. Final summary
log_output("\n" + "="*80)
log_output("TRAINING COMPLETE!")
log_output("="*80)
log_output(f"\nTraining duration: {training_duration}")
log_output(f"Output files:")
log_output(f"  - Model: {OUTPUT_DIR}")
log_output(f"  - Training log: {TRAINING_OUTPUT_FILE}")
log_output(f"  - Metrics JSON: {TRAINING_METRICS_FILE}")

log_output("\nTo use the fine-tuned model for inference:")
log_output(f"""
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="{OUTPUT_DIR}",
    device=-1,  # or 0 for GPU
    truncation=True
)

results = pipe(["Your text here"])
print(results)
""")

log_output(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_output("="*80)
