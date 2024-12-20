# scripts/load_dataset.py

import os
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd

def load_csv_files(processed_data_dir):
    """
    Loads train and validation CSV files into Pandas DataFrames.

    Parameters:
    - processed_data_dir (str): Path to the directory containing train.csv and validation.csv.

    Returns:
    - train_df (pd.DataFrame): Training DataFrame.
    - val_df (pd.DataFrame): Validation DataFrame.
    """
    train_path = os.path.join(processed_data_dir, 'train.csv')
    val_path = os.path.join(processed_data_dir, 'validation.csv')

    # Load CSV files into DataFrames
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    print(f"Loaded {len(train_df)} training samples from {train_path}.")
    print(f"Loaded {len(val_df)} validation samples from {val_path}.")

    return train_df, val_df

def create_dataset_dict(train_df, val_df):
    """
    Creates a Hugging Face DatasetDict from Pandas DataFrames.

    Parameters:
    - train_df (pd.DataFrame): Training DataFrame.
    - val_df (pd.DataFrame): Validation DataFrame.

    Returns:
    - dataset_dict (DatasetDict): Hugging Face DatasetDict with 'train' and 'validation' splits.
    """
    # Convert Pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Create a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    print("DatasetDict created with 'train' and 'validation' splits.")
    return dataset_dict

def main():
    """
    Main function to load datasets into Hugging Face's datasets library.
    """
    # Define the path to the processed data directory
    processed_data_dir = os.path.join('..', 'data', 'processed')

    # Load CSV files
    train_df, val_df = load_csv_files(processed_data_dir)

    # Create DatasetDict
    dataset_dict = create_dataset_dict(train_df, val_df)

    # Optional: Save the DatasetDict to disk for faster loading in the future
    # dataset_dict.save_to_disk(os.path.join('..', 'data', 'processed', 'hf_dataset'))
    # print("DatasetDict saved to disk.")

    # Inspect the dataset (e.g., view the first example in the training set)
    print("\nSample Training Example:")
    print(dataset_dict['train'][0])

    print("\nSample Validation Example:")
    print(dataset_dict['validation'][0])

if __name__ == "__main__":
    main()
