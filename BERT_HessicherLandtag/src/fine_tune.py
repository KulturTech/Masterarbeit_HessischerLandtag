import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import pandas as pd
import numpy as np

# Configuration
MODEL_NAME = "Hate-speech-CNERG/dehatebert-mono-german"
OUTPUT_DIR = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\fine_tuned_model"
TRAINING_DATA_PATH = r"c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\training\labeled_data.parquet"
MAX_LENGTH = 512
BATCH_SIZE = 8  # Adjust based on your GPU/CPU memory
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42

print("="*80)
print("BERT FINE-TUNING SCRIPT")
print("="*80)

# 1. Check device availability
if torch.cuda.is_available():
    device = 0
    print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    print("\nNo GPU available. Using CPU for training.")
    print("Warning: Training on CPU will be significantly slower.")

# 2. Load your labeled training data
print(f"\nLoading training data from: {TRAINING_DATA_PATH}")
try:
    df = pd.read_parquet(TRAINING_DATA_PATH)
    print(f"Loaded {len(df)} labeled examples")
    print(f"Columns: {df.columns.tolist()}")

    # Assuming your data has 'text' and 'label' columns
    # Adjust column names as needed
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must have 'text' and 'label' columns")

    print(f"\nLabel distribution:")
    print(df['label'].value_counts())

except FileNotFoundError:
    print(f"Error: Could not find training data at {TRAINING_DATA_PATH}")
    print("\nPlease create a labeled dataset with 'text' and 'label' columns.")
    print("Example format:")
    print("  text                          | label")
    print("  'This is hate speech'         | 'HATE'")
    print("  'This is normal text'         | 'NON-HATE'")
    exit(1)

# 3. Prepare label mapping
unique_labels = df['label'].unique()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(unique_labels)

print(f"\nLabel mapping:")
for label, idx in label2id.items():
    print(f"  {label} -> {idx}")

# Convert labels to numeric IDs
df['label_id'] = df['label'].map(label2id)

# 4. Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label_id'].tolist(),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df['label_id']  # Maintain label distribution
)

print(f"\nDataset split:")
print(f"  Training samples: {len(train_texts)}")
print(f"  Validation samples: {len(val_texts)}")

# 5. Load tokenizer and model
print(f"\nLoading model and tokenizer: {MODEL_NAME}")
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

print("\nTokenizing datasets...")
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
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,  # Keep only 2 best checkpoints
    report_to="none",  # Disable wandb/tensorboard for simplicity
)

print("\nTraining configuration:")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Max sequence length: {MAX_LENGTH}")

# 9. Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 10. Train the model
print("\n" + "="*80)
print("STARTING TRAINING")
print("="*80 + "\n")

trainer.train()

# 11. Evaluate on validation set
print("\n" + "="*80)
print("FINAL EVALUATION")
print("="*80 + "\n")

eval_results = trainer.evaluate()
print("Validation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

# 12. Save the fine-tuned model
print(f"\nSaving fine-tuned model to: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"\nYour fine-tuned model is saved at: {OUTPUT_DIR}")
print("\nTo use it for inference:")
print(f"""
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
