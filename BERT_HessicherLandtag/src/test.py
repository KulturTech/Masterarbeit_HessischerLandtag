import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm

# 1. Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    device = 0  # Use the first GPU
    print("CUDA-enabled GPU is available. Using GPU for inference.")
else:
    device = -1  # Use CPU
    print("No CUDA-enabled GPU found. Using CPU for inference.")

# 2. Set the desired batch_size
# For CPU: smaller batch sizes (8-16) work better
# For GPU: larger batch sizes (32-64) are more efficient
batch_size = 8  # Reduced for CPU processing
print(f"Setting batch size to: {batch_size}")

# 3. Re-initialize the pipeline with the new batch_size and device
# The model 'Hate-speech-CNERG/dehatebert-mono-german' is used for text-classification
pipe = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-german", device=device, batch_size=batch_size)
print("Text classification pipeline re-initialized with new batch size.")

df = pd.read_parquet('BERT_HessicherLandtag/Data/prep_v1/all_docs_clean.parquet')
print("Parquet file loaded successfully.")
print(f"Total documents: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# 4. Apply the model to all texts
print("\nApplying hate speech classification to all documents...")
texts = df['text'].tolist()

# Process in batches using the pipeline with progress bar
print(f"Processing {len(texts)} documents in batches of {batch_size}...")
results = []
for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
    batch = texts[i:i+batch_size]
    batch_results = pipe(batch)
    results.extend(batch_results)

# 5. Add predictions to dataframe
df['label'] = [result['label'] for result in results]
df['score'] = [result['score'] for result in results]

print("\nClassification complete!")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"\nFirst 5 results:")
print(df[['doc_id', 'label', 'score']].head())

# 6. Save results
output_path = 'BERT_HessicherLandtag/Data/prep_v1/all_docs_classified.parquet'
df.to_parquet(output_path)
print(f"\nResults saved to: {output_path}")