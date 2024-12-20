# scripts/preprocess_tokenize.py

import os
from datasets import load_dataset, Dataset, DatasetDict
from transformers import T5Tokenizer
import torch

def load_dataset_dict():
    """
    Loads the DatasetDict from the CSV files.
    """
    data_files = {
        "train": "../data/processed/train.csv",
        "validation": "../data/processed/validation.csv"
    }

    # Load the dataset
    dataset = load_dataset('csv', data_files=data_files)
    return dataset

def filter_non_null(example):
    """
    Filters out examples where 'Early_Bio' or 'Later_Bio' is None.
    """
    return example['Early_Bio'] is not None and example['Later_Bio'] is not None

def preprocess_and_tokenize(dataset, tokenizer, input_column='Early_Bio', target_column='Later_Bio', max_input_length=128, max_target_length=128):
    """
    Preprocesses and tokenizes the dataset.

    Parameters:
    - dataset (DatasetDict): The dataset to preprocess.
    - tokenizer (T5Tokenizer): The tokenizer to use.
    - input_column (str): The name of the input text column.
    - target_column (str): The name of the target text column.
    - max_input_length (int): Maximum length for input tokens.
    - max_target_length (int): Maximum length for target tokens.

    Returns:
    - tokenized_datasets (DatasetDict): The tokenized dataset.
    """
    # Filter out examples with None values
    print("Filtering out examples with None values...")
    dataset = dataset.filter(filter_non_null)
    print(f"Number of training samples after filtering: {len(dataset['train'])}")
    print(f"Number of validation samples after filtering: {len(dataset['validation'])}")

    def preprocess_function(examples):
        # Add task-specific prefix to inputs
        inputs = ["predict future bio: " + bio for bio in examples[input_column]]
        targets = [bio for bio in examples[target_column]]

        # Tokenize inputs
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            padding='max_length',
            truncation=True
        )

        # Tokenize targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding='max_length',
                truncation=True
            )

        # Replace padding token id's of the labels by -100 to ignore them in loss computation
        labels["input_ids"] = [
            [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
            for label_seq in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the preprocessing function to the dataset with a progress bar
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[input_column, target_column],  # Remove original text columns
        desc="Tokenizing dataset"  # Description for the progress bar
    )

    return tokenized_datasets

def main():
    """
    Main function to preprocess and tokenize the dataset.
    """
    # Load the DatasetDict
    print("Loading the dataset...")
    dataset = load_dataset_dict()

    # Initialize the tokenizer
    print("Initializing the T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Preprocess and tokenize the dataset
    print("Preprocessing and tokenizing the dataset...")
    tokenized_datasets = preprocess_and_tokenize(dataset, tokenizer)

    # Save the tokenized dataset to disk for future use (optional)
    # Uncomment the following lines if you wish to save the dataset
    tokenized_datasets.save_to_disk("../data/processed/hf_tokenized_dataset")
    print("Tokenized DatasetDict saved to disk.")

    # Inspect a tokenized example
    print("\nSample Tokenized Training Example:")
    print(tokenized_datasets['train'][0])

    print("\nSample Tokenized Validation Example:")
    print(tokenized_datasets['validation'][0])

if __name__ == "__main__":
    main()
