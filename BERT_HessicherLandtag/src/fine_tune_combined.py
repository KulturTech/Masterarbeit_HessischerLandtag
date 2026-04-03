import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

BASE_DIR         = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag"
MODEL_DIR        = BASE_DIR + r"\fine_tuned_model_cv\best_model"   # weiteres Fine-Tuning auf bestehendem Modell
TRAINING_DATA    = BASE_DIR + r"\Data\training\labeled_data_combined.csv"
OUTPUT_DIR       = BASE_DIR + r"\fine_tuned_model_combined"

MAX_LENGTH   = 512
BATCH_SIZE   = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS   = 3
TEST_SIZE    = 0.2
RANDOM_STATE = 42

print("=" * 70)
print("FINE-TUNING mit kombiniertem Datensatz")
print("=" * 70)

device = 0 if torch.cuda.is_available() else -1
print(f"Device: {'GPU' if device == 0 else 'CPU'}")

# Daten laden
df = pd.read_csv(TRAINING_DATA)
print(f"\nTrainingsdaten: {len(df)} Samples")
print(df['label'].value_counts())

# Label-Mapping konsistent mit bestehendem Modell
label2id = {"NON_HATE": 0, "HATE": 1}
id2label  = {0: "NON_HATE", 1: "HATE"}
df['label_id'] = df['label'].map(label2id)

# Train/Val Split (stratifiziert)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(),
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label_id']
)
print(f"\nTrain: {len(train_texts)} | Val: {len(val_texts)}")

# Tokenizer & Modell vom bestehenden Checkpoint laden
print(f"\nLade Modell von: {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

# Tokenisierung
def tokenize(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

train_ds = Dataset.from_dict({'text': train_texts, 'label': train_labels}).map(tokenize, batched=True)
val_ds   = Dataset.from_dict({'text': val_texts,   'label': val_labels}).map(tokenize, batched=True)

# Metriken
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1, 'precision': p, 'recall': r}

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=train_ds, eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("\n" + "=" * 70)
print("TRAINING STARTET")
print("=" * 70)
trainer.train()

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)
results = trainer.evaluate()
for k, v in results.items():
    print(f"  {k}: {v:.4f}")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n[OK] Modell gespeichert: {OUTPUT_DIR}")
