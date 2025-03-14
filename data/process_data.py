import os
import pandas as pd
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config

def process_data():
    # Get the configuration values from config.py
    config = get_config()
    dataset_dir = config['processed_dataset_dir']
    threshold = config['ticker_threshold']

    # Initialize an empty list to hold the DataFrames
    dfs = []

    # Walk through the subfolders in the dataset directory
    for subdir, _, files in os.walk(dataset_dir):
        # Check if the folder contains the 'dataset.csv' file
        if 'dataset.csv' in files:
            # Extract the ticker from the folder name (you can adjust this if needed)
            ticker = os.path.basename(subdir)  # Assuming the folder name is the ticker symbol
            
            # Read the CSV file and parse the 'Date' column as datetime
            df = pd.read_csv(os.path.join(subdir, 'dataset.csv'), parse_dates=["Date"], index_col=0)
            
            # Check if the DataFrame has more rows than the threshold
            if len(df) > threshold:
                # Rename columns to include the ticker (optional)
                df.columns = [f'{ticker}_{col}' for col in df.columns]
                
                # Append the DataFrame to the list
                dfs.append(df)

    # Concatenate the DataFrames on the 'Date' index
    result = pd.concat(dfs, axis=1, join='inner')  # 'inner' join to keep only common dates

    # Get the current date for the filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Construct the filename with the current date
    filename = f"data/dataset/dataset_{current_date}.csv"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the result to the dynamic filename
    result.to_csv(filename)

    # Print confirmation and the result
    print(f"CSV file has been saved as '{filename}'. Here is a preview:")
    print(result.head())

if __name__ == "__main__":
    process_data()