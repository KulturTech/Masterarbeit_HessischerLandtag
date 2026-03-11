import pandas as pd
import os

def identify_false_negatives():
    # Load the immigrant_docs_classified.parquet file
    immigrant_file = r'BERT_HessicherLandtag\Data\prep_v1\immigrant_docs_classified.parquet'
    imm_df = pd.read_parquet(immigrant_file)

    print('Loaded immigrant_docs_classified.parquet')
    print(f'Total documents: {len(imm_df)}')
    print('Columns:', list(imm_df.columns))
    print()

    # Check unique values in hate_label
    print('Hate label distribution:')
    print(imm_df['hate_label'].value_counts())
    print()

    # Filter NON_HATE predictions
    non_hate_df = imm_df[imm_df['hate_label'] == 'NON_HATE']
    print(f'NON_HATE predictions: {len(non_hate_df)}')

    # Sort by hate_score ascending (lowest confidence first)
    non_hate_sorted = non_hate_df.sort_values('hate_score').head(20)
    print('Top 20 lowest confidence NON_HATE predictions:')
    print(non_hate_sorted[['doc_id', 'hate_label', 'hate_score']].head(10))
    print()

    # Summary statistics
    print('Summary statistics of prediction scores:')
    print(imm_df['hate_score'].describe())
    print()

    # Load manually labeled HATE documents
    labeled_file = r'BERT_HessicherLandtag\Data\training\labeled_data.parquet'
    if os.path.exists(labeled_file):
        labeled_df = pd.read_parquet(labeled_file)
        print('Loaded labeled_data.parquet')
        print(f'Manually labeled documents: {len(labeled_df)}')
        print('Label distribution in labeled data:')
        print(labeled_df['hate_label'].value_counts())
    else:
        print('labeled_data.parquet not found')
        labeled_df = None

    # Save results to CSV
    output_dir = r'BERT_HessicherLandtag\Data\evaluation'
    os.makedirs(output_dir, exist_ok=True)

    # Save the 20 lowest confidence NON_HATE predictions
    output_file = os.path.join(output_dir, 'false_negatives_candidates.csv')
    non_hate_sorted.to_csv(output_file, index=False)
    print(f'Saved false negatives candidates to: {output_file}')

    # Summary statistics file
    stats_file = os.path.join(output_dir, 'prediction_statistics.csv')
    stats_df = imm_df['hate_score'].describe().to_frame().T
    stats_df.to_csv(stats_file, index=False)
    print(f'Saved prediction statistics to: {stats_file}')

    return non_hate_sorted, imm_df, labeled_df

if __name__ == "__main__":
    identify_false_negatives()