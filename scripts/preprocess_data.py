import pandas as pd
import os
import re

def clean_text(text):
    """
    Cleans the input text by removing URLs, special characters, and extra spaces.
    """
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_file(input_path, output_path):
    """
    Reads a CSV file, cleans the Twitter bios, and saves the processed data.
    """
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # Assuming the CSV has 'Early_Bio' and 'Later_Bio' columns
    df['Early_Bio'] = df['Early_Bio'].astype(str).apply(clean_text)
    df['Later_Bio'] = df['Later_Bio'].astype(str).apply(clean_text)
    
    # Optionally, handle missing values
    df.dropna(subset=['Early_Bio', 'Later_Bio'], how='all', inplace=True)
    
    # Save the processed data
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    raw_data_dir = os.path.join('..', 'data', 'raw')
    processed_data_dir = os.path.join('..', 'data', 'processed')
    
    # Create processed_data_dir if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Iterate through all CSV files in raw_data_dir
    for filename in os.listdir(raw_data_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(raw_data_dir, filename)
            output_file = os.path.join(processed_data_dir, f"processed_{filename}")
            preprocess_file(input_file, output_file)

