import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import re

# 1. Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    device = 0
    print("CUDA-enabled GPU is available. Using GPU for inference.")
else:
    device = -1
    print("No CUDA-enabled GPU found. Using CPU for inference.")

# 2. Set the desired batch_size
batch_size = 8
print(f"Setting batch size to: {batch_size}")

# 3. Initialize the hate speech detection pipeline
print("\nInitializing hate speech detection model...")
pipe = pipeline(
    "text-classification",
    model="Hate-speech-CNERG/dehatebert-mono-german",
    device=device,
    batch_size=batch_size,
    truncation=True
)
print("Pipeline initialized.")

# 4. Define immigrant/migration-related keywords (German)
# These keywords help identify text discussing immigrants/migration
IMMIGRANT_KEYWORDS = [
    # Direct terms
    r'\b(immigrant|immigranten|einwanderer|einwanderung|zuwander|zuwanderung)\w*\b',
    r'\b(migrant|migranten|migration)\w*\b',
    r'\b(flüchtling|flüchtlinge|asyl|asylbewerber|asylsuchende)\w*\b',
    r'\b(ausländer|ausländisch)\w*\b',
    r'\b(geflüchtete|schutzsuchende)\w*\b',

    # Related terms
    r'\b(integration|integrationspolitik)\w*\b',
    r'\b(abschiebung|abgeschoben|rückführung)\w*\b',
    r'\b(aufenthaltsrecht|aufenthaltsstatus|bleiberecht)\w*\b',
    r'\b(grenzsicherung|grenzschutz|grenzkontrollen)\w*\b',

    # Origin-based references
    r'\b(syrer|syrien|afghanistan|afrika|afrikanisch)\w*\b',
    r'\b(nordafrika|nahost|balkan)\w*\b',

    # Policy terms
    r'\b(asylpolitik|migrationspolitik|ausländerpolitik)\w*\b',
    r'\b(aufnahme|verteilung|kontingent)\w*\b',
]

# Compile regex patterns for efficiency
immigrant_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in IMMIGRANT_KEYWORDS]

def contains_immigrant_reference(text):
    """Check if text contains references to immigrants/migration"""
    if not isinstance(text, str):
        return False

    for pattern in immigrant_patterns:
        if pattern.search(text):
            return True
    return False

def extract_immigrant_keywords(text):
    """Extract which immigrant-related keywords appear in the text"""
    if not isinstance(text, str):
        return []

    found_keywords = []
    for pattern in immigrant_patterns:
        matches = pattern.findall(text)
        if matches:
            found_keywords.extend(matches)
    return list(set(found_keywords))  # Remove duplicates

# 5. Load your data
print("\n" + "="*80)
print("DETECTING HATE SPEECH AGAINST IMMIGRANTS")
print("="*80)

data_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\all_docs_clean.parquet'
print(f"\nLoading data from: {data_path}")
df = pd.read_parquet(data_path)
print(f"Total documents: {len(df)}")

# 6. First pass: Filter for immigrant-related content
print("\nStep 1: Filtering for immigrant-related content...")
df['mentions_immigrants'] = df['text'].apply(contains_immigrant_reference)
df['immigrant_keywords'] = df['text'].apply(extract_immigrant_keywords)

immigrant_docs = df[df['mentions_immigrants']].copy()
print(f"Found {len(immigrant_docs)} documents mentioning immigrants/migration ({len(immigrant_docs)/len(df)*100:.2f}%)")

# 7. Second pass: Run hate speech detection on immigrant-related documents
if len(immigrant_docs) > 0:
    print("\nStep 2: Running hate speech detection on immigrant-related documents...")
    texts = immigrant_docs['text'].tolist()

    # Process in batches with progress bar
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying"):
        batch = texts[i:i+batch_size]
        batch_results = pipe(batch)
        results.extend(batch_results)

    # Add predictions to dataframe
    immigrant_docs['hate_label'] = [result['label'] for result in results]
    immigrant_docs['hate_score'] = [result['score'] for result in results]

    # 8. Filter for hate speech
    hate_against_immigrants = immigrant_docs[immigrant_docs['hate_label'] == 'HATE'].copy()

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTotal documents analyzed: {len(df)}")
    print(f"Documents mentioning immigrants: {len(immigrant_docs)} ({len(immigrant_docs)/len(df)*100:.2f}%)")
    print(f"Hate speech in immigrant-related docs: {len(hate_against_immigrants)} ({len(hate_against_immigrants)/len(immigrant_docs)*100:.2f}%)")
    print(f"Overall hate speech against immigrants: {len(hate_against_immigrants)} ({len(hate_against_immigrants)/len(df)*100:.2f}%)")

    # 9. Show statistics
    print("\n" + "="*80)
    print("HATE SPEECH STATISTICS")
    print("="*80)

    print("\nLabel distribution in immigrant-related documents:")
    print(immigrant_docs['hate_label'].value_counts())

    if len(hate_against_immigrants) > 0:
        print(f"\nHate speech confidence scores (mean): {hate_against_immigrants['hate_score'].mean():.4f}")
        print(f"Hate speech confidence scores (median): {hate_against_immigrants['hate_score'].median():.4f}")
        print(f"Hate speech confidence scores (min): {hate_against_immigrants['hate_score'].min():.4f}")
        print(f"Hate speech confidence scores (max): {hate_against_immigrants['hate_score'].max():.4f}")

        # Most common immigrant-related keywords in hate speech
        all_keywords = []
        for keywords in hate_against_immigrants['immigrant_keywords']:
            all_keywords.extend(keywords)

        if all_keywords:
            from collections import Counter
            keyword_counts = Counter(all_keywords)
            print("\nMost common immigrant-related terms in hate speech:")
            for keyword, count in keyword_counts.most_common(10):
                print(f"  {keyword}: {count}")

        print(f"\nSample hate speech against immigrants (top 5 by confidence):")
        print("="*80)
        top_hate = hate_against_immigrants.nlargest(5, 'hate_score')
        for idx, row in top_hate.iterrows():
            print(f"\nDoc ID: {row.get('doc_id', idx)}")
            print(f"Confidence: {row['hate_score']:.4f}")
            print(f"Keywords: {', '.join(row['immigrant_keywords'][:5])}")
            # Show first 300 characters of text
            text_preview = row['text'][:300] + "..." if len(row['text']) > 300 else row['text']
            print(f"Text: {text_preview}")
            print("-"*80)

    # 10. Save results
    output_path_all = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\immigrant_docs_classified.parquet'
    immigrant_docs.to_parquet(output_path_all)
    print(f"\nAll immigrant-related documents saved to: {output_path_all}")

    if len(hate_against_immigrants) > 0:
        output_path_hate = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\hate_speech_against_immigrants.parquet'
        hate_against_immigrants.to_parquet(output_path_hate)
        print(f"Hate speech against immigrants saved to: {output_path_hate}")

        # Also save as CSV for easy viewing
        output_csv = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\hate_speech_against_immigrants.csv'
        hate_against_immigrants.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"Hate speech against immigrants saved to: {output_csv}")

    # 11. Create summary statistics file
    summary = {
        'total_documents': len(df),
        'immigrant_related_docs': len(immigrant_docs),
        'immigrant_related_percentage': len(immigrant_docs)/len(df)*100,
        'hate_speech_count': len(hate_against_immigrants),
        'hate_speech_percentage_of_immigrant_docs': len(hate_against_immigrants)/len(immigrant_docs)*100 if len(immigrant_docs) > 0 else 0,
        'hate_speech_percentage_of_all_docs': len(hate_against_immigrants)/len(df)*100,
        'label_distribution': immigrant_docs['hate_label'].value_counts().to_dict()
    }

    import json
    summary_path = r'c:\Users\gsera\OneDrive\Desktop\Masterarbeit\Masterarbeit_HessischerLandtag\BERT_HessicherLandtag\Data\prep_v1\immigrant_hate_speech_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary statistics saved to: {summary_path}")

else:
    print("\nNo documents found mentioning immigrants/migration.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
