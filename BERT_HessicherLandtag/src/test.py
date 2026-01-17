import torch
from transformers import pipeline
import pandas as pd

# 1. Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    device = 0  # Use the first GPU
    print("CUDA-enabled GPU is available. Using GPU for inference.")
else:
    device = -1  # Use CPU
    print("No CUDA-enabled GPU found. Using CPU for inference.")

# 2. Set the desired batch_size
# You can start with values like 16, 32, 64, and increase/decrease as needed
batch_size = 32
print(f"Setting batch size to: {batch_size}")

# 3. Re-initialize the pipeline with the new batch_size and device
# The model 'Hate-speech-CNERG/dehatebert-mono-german' is used for text-classification
pipe = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-german", device=device, batch_size=batch_size)
print("Text classification pipeline re-initialized with new batch size.")

df = pd.read_parquet('BERT_HessicherLandtag/Data/prep_v1/all_docs_clean.parquet')
print("Parquet file loaded successfully. First 5 rows:")
display(df.head())