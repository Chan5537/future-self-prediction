# scripts/apply_similarity.py

import pandas as pd
import numpy as np
from sentence_similarity import SimilarityCalculator
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings

def compute_similarity_row(args):
    """
    Worker function to compute similarity scores for a single row.
    
    Parameters:
    - args (tuple): A tuple containing the row and the similarity calculator instance.
    
    Returns:
    - dict: A dictionary of similarity scores.
    """
    row, sim_calc = args
    bio_2015 = row['Early_Bio']
    bio_2020 = row['Later_Bio']
    
    if pd.notna(bio_2015) and pd.notna(bio_2020):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress BLEU warnings
            return sim_calc.calculate_all(bio_2015, bio_2020)
    else:
        return {
            'Jaccard': np.nan,
            'BLEU': np.nan,
            'ROUGE1': np.nan,
            'ROUGE2': np.nan,
            'ROUGEL': np.nan,
            'Cosine_Similarity': np.nan
        }

def main():
    """
    Main function to apply similarity metrics to processed bio data.
    """
    # Define paths
    processed_data_dir = '../data/processed/'
    csv_files = [
        'processed_PTS_Bio_Pairs_01.csv',
        'processed_PTS_Bio_Pairs_02.csv',
        'processed_PTS_Bio_Pairs_03.csv'
    ]
    output_suffix = '_with_similarity.csv'
    
    # Initialize the similarity calculator
    sim_calc = SimilarityCalculator()
    
    # Determine the number of processes
    num_processes = cpu_count()
    
    # Iterate through each CSV file
    for file in csv_files:
        input_path = os.path.join(processed_data_dir, file)
        output_file = file.replace('.csv', output_suffix)
        output_path = os.path.join(processed_data_dir, output_file)
        
        # Load the dataset
        df = pd.read_csv(input_path)
        
        # Prepare arguments for multiprocessing
        args = [(row, sim_calc) for _, row in df.iterrows()]
        
        # Initialize lists to store similarity scores
        jaccard_scores = []
        bleu_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougel_scores = []
        cosine_similarities = []
        
        # Use multiprocessing Pool with progress bar
        with Pool(processes=num_processes) as pool:
            for scores in tqdm(pool.imap_unordered(compute_similarity_row, args), total=len(args), desc=f'Processing {file}'):
                jaccard_scores.append(scores['Jaccard'])
                bleu_scores.append(scores['BLEU'])
                rouge1_scores.append(scores['ROUGE1'])
                rouge2_scores.append(scores['ROUGE2'])
                rougel_scores.append(scores['ROUGEL'])
                cosine_similarities.append(scores['Cosine_Similarity'])
        
        # Assign the similarity scores to new columns
        df['Jaccard'] = jaccard_scores
        df['BLEU'] = bleu_scores
        df['ROUGE1'] = rouge1_scores
        df['ROUGE2'] = rouge2_scores
        df['ROUGEL'] = rougel_scores
        df['Cosine_Similarity'] = cosine_similarities
        
        # Save the updated DataFrame
        df.to_csv(output_path, index=False)
        print(f"Similarity scores added and saved to {output_path}")

if __name__ == "__main__":
    main()
