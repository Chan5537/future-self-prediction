# scripts/split_dataset.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_concatenate_csv(processed_data_dir):
    """
    Loads all CSV files from the processed_data_dir and concatenates them into a single DataFrame.

    Parameters:
    - processed_data_dir (str): Path to the directory containing processed CSV files.

    Returns:
    - pd.DataFrame: Concatenated DataFrame containing all bios.
    """
    csv_files = [file for file in os.listdir(processed_data_dir) if file.endswith('.csv')]
    df_list = []

    for file in csv_files:
        file_path = os.path.join(processed_data_dir, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    concatenated_df = pd.concat(df_list, ignore_index=True)
    return concatenated_df

def split_dataset(df, test_size=0.1, random_state=42):
    """
    Splits the DataFrame into training and validation sets.

    Parameters:
    - df (pd.DataFrame): The DataFrame to split.
    - test_size (float): Proportion of the dataset to include in the validation set.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - pd.DataFrame, pd.DataFrame: Training and validation DataFrames.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df

def save_split_datasets(train_df, val_df, processed_data_dir):
    """
    Saves the training and validation DataFrames as CSV files.

    Parameters:
    - train_df (pd.DataFrame): Training DataFrame.
    - val_df (pd.DataFrame): Validation DataFrame.
    - processed_data_dir (str): Directory to save the split CSV files.
    """
    train_path = os.path.join(processed_data_dir, 'train.csv')
    val_path = os.path.join(processed_data_dir, 'validation.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Training dataset saved to {train_path} ({len(train_df)} samples).")
    print(f"Validation dataset saved to {val_path} ({len(val_df)} samples).")

def main():
    """
    Main function to execute the dataset splitting process.
    """
    # Define the path to the processed data directory
    processed_data_dir = os.path.join('..', 'data', 'processed')

    # Load and concatenate all CSV files
    print("Loading and concatenating CSV files...")
    concatenated_df = load_and_concatenate_csv(processed_data_dir)
    print(f"Total samples loaded: {len(concatenated_df)}")

    # Split the dataset
    print("Splitting the dataset into training and validation sets...")
    train_df, val_df = split_dataset(concatenated_df, test_size=0.1, random_state=42)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Save the split datasets
    print("Saving the split datasets...")
    save_split_datasets(train_df, val_df, processed_data_dir)
    print("Dataset splitting completed successfully.")

if __name__ == "__main__":
    main()
