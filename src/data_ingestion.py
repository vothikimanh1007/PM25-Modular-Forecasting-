# src/data_ingestion.py
import pandas as pd
import os
import urllib.request

def load_local_data(filepath):
    """
    Loads PM2.5 data from a local CSV file (simulating IoT monitoring stations).
    """
    print(f"Attempting to load local data from: {filepath}")
    if not os.path.exists(filepath):
         raise FileNotFoundError(f"Data file not found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Successfully loaded {len(df)} records.")
    return df

def download_benchmark_data(output_path="data/raw/pollution.csv"):
    """
    Downloads the Beijing PM2.5 benchmark dataset if it doesn't already exist.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pollution.csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(output_path):
        print(f"Downloading benchmark dataset from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print("Download complete.")
    else:
        print("Benchmark dataset already exists locally.")
        
    return load_local_data(output_path)

if __name__ == "__main__":
    # Test the ingestion module
    df = download_benchmark_data()
    print(df.head())