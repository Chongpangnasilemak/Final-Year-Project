import requests
import pandas as pd 
import numpy as np 
from functools import reduce
import os
from dotenv import load_dotenv
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_config

# Load environment variables from .env file
load_dotenv()

API_KEY =  os.getenv("API_KEY")

def get_adjusted_daily(ticker, config, output_size='full', api_key=API_KEY):

    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&outputsize={output_size}&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None  # Return None to indicate failure
    
    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Time Series (Daily)' exists in the response
    if 'Time Series (Daily)' not in data:
        print(f"Error: 'Time Series (Daily)' not found for {ticker}.")
        return None

    # Extract the relevant time series data
    price_data = data['Time Series (Daily)']

    # Convert the data into a list of tuples and create a DataFrame
    price_list = [
        (date, 
         float(info['1. open']), 
         float(info['2. high']), 
         float(info['3. low']),
         float(info['4. close']), 
         float(info['5. adjusted close']))
        for date, info in price_data.items()
    ]
    
    df = pd.DataFrame(price_list, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted Close'])

    # Drop open, high, low, close since not adjusted for stock splits
    df = df.drop(columns=['Open','High','Low','Close'],axis=1)

    # Convert 'Date' column to datetime and set it as the index
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataframe to a CSV file
    output_file = os.path.join(output_dir, 'TIME_SERIES_DAILY.csv')
    df.to_csv(output_file)
    print(f"Data for {ticker} saved to {output_file}")

    return df

def get_fx_daily(ticker, config, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&outputsize=full&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching FX data: {e}")
        return None  # Return None to indicate failure
    
    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Time Series FX (Daily)' exists in the response
    if 'Time Series FX (Daily)' not in data:
        print("Error: 'Time Series FX (Daily)' not found.")
        return None

    # Extract the relevant FX data
    fx_data = data['Time Series FX (Daily)']

    # Convert the data into a DataFrame
    df = pd.DataFrame.from_dict(fx_data, orient="index")
    
    # Rename columns and convert the index to datetime
    df.columns = ['Open', 'High', 'Low', 'Close']
    df.index = pd.to_datetime(df.index)
    
    # Convert all columns to float type
    df = df.astype(float)
    
    df.reset_index(inplace=True)  # Ensure Date is a column
    df.rename(columns={'index': 'Date'}, inplace=True)
    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataframe to a CSV file
    output_file = os.path.join(output_dir, 'EUR_USD_FX_DAILY.csv')
    df.to_csv(output_file)
    print(f"FX data saved to {output_file}")

    return df

def get_crypto_daily(ticker, config, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=USD&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching crypto data: {e}")
        return None  # Return None to indicate failure
    
    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Time Series (Digital Currency Daily)' exists in the response
    if 'Time Series (Digital Currency Daily)' not in data:
        print("Error: 'Time Series (Digital Currency Daily)' not found.")
        return None

    # Extract the relevant crypto data
    crypto_data = data['Time Series (Digital Currency Daily)']

    # Convert the data into a DataFrame
    df = pd.DataFrame.from_dict(crypto_data, orient="index")
    
    # Rename columns and convert the index to datetime
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)
    df.reset_index(inplace=True)  # Ensure Date is a column
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Convert all columns to float type

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the dataframe to a CSV file
    output_file = os.path.join(output_dir, 'BTC_DAILY.csv')
    df.to_csv(output_file)
    print(f"Crypto data saved to {output_file}")

    return df

def get_fed_fund_daily(ticker, config, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Federal Funds Rate data: {e}")
        return None  # Return None to indicate failure
    
    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'data' exists in the response
    if 'data' not in data:
        print("Error: 'data' not found in the response.")
        return None

    # Extract the Federal Funds Rate data
    fed_rate_data = data['data']

    # Convert the data into a DataFrame
    df = pd.DataFrame(fed_rate_data)

    # Rename columns and handle the Date column
    df.rename(columns={'date': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Set the Date column as the index and reset the index
    df.set_index('Date', inplace=True)
    df = df.rename_axis("Date").reset_index()

    # Rename the column for the Federal Funds Rate value
    df.rename(columns={'value': 'Effective_Fed_Rate'}, inplace=True)

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, 'FEDERAL_FUNDS_RATE.csv')
    df.to_csv(output_file,index=False)
    print(f"Federal Funds Rate data saved to {output_file}")

    return df

def get_sma_daily(ticker, config, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=SMA&symbol={ticker}&interval=daily&time_period=10&series_type=open&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching SMA data for {ticker}: {e}")
        return None  # Return None to indicate failure
    
    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Technical Analysis: SMA' exists in the response
    if 'Technical Analysis: SMA' not in data:
        print(f"Error: 'Technical Analysis: SMA' not found for {ticker}.")
        return None

    # Extract the SMA data
    sma_data = data['Technical Analysis: SMA']

    # Convert the data into a list of tuples (Date, SMA) and create the DataFrame
    sma_list = [(date, float(sma_info['SMA'])) for date, sma_info in sma_data.items()]

    # Create DataFrame from the SMA data
    df = pd.DataFrame(sma_list, columns=['Date', 'SMA'])

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Set the 'Date' column as the index
    df.set_index('Date', inplace=True)
    df = df.rename_axis("Date").reset_index()

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, 'SMA.csv')
    df.to_csv(output_file,index=False)
    print(f"SMA data for {ticker} saved to {output_file}")

    return df

def get_ema_daily(ticker, config, time_period=10, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=EMA&symbol={ticker}&interval=daily&time_period={time_period}&series_type=open&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching EMA data for {ticker}: {e}")
        return None  # Return None to indicate failure

    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Technical Analysis: EMA' exists in the response
    if 'Technical Analysis: EMA' not in data:
        print(f"Error: 'Technical Analysis: EMA' not found for {ticker}.")
        return None

    # Extract the EMA data
    ema_data = data['Technical Analysis: EMA']

    # Convert the data into a DataFrame
    df = pd.DataFrame.from_dict(ema_data, orient="index")
    
    # Convert the index to datetime format
    df.index = pd.to_datetime(df.index)
    
    # Convert the 'EMA' column to float type
    df["EMA"] = df["EMA"].astype(float)
    
    # Rename the index and reset to a regular column
    df = df.rename_axis("Date").reset_index()

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, 'EMA.csv')
    df.to_csv(output_file, index=False)  # Set index=False to avoid saving the index as a separate column
    
    print(f"EMA data for {ticker} saved to {output_file}")

    return df

def get_macd_daily(ticker, config, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=MACD&symbol={ticker}&interval=daily&series_type=open&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching MACD data for {ticker}: {e}")
        return None  # Return None to indicate failure

    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Technical Analysis: MACD' exists in the response
    if 'Technical Analysis: MACD' not in data:
        print(f"Error: 'Technical Analysis: MACD' not found for {ticker}.")
        return None

    # Extract the MACD data
    macd_data = data['Technical Analysis: MACD']

    # Convert the data into a DataFrame
    df = pd.DataFrame.from_dict(macd_data, orient="index")
    
    # Convert the index to datetime format
    df.index = pd.to_datetime(df.index)
    
    # Convert the DataFrame to float type
    df = df.astype(float)
    
    # Rename the index and reset to a regular column
    df = df.rename_axis("Date").reset_index()

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, 'MACD.csv')
    df.to_csv(output_file, index=False)  # Set index=False to avoid saving the index as a separate column
    
    print(f"MACD data for {ticker} saved to {output_file}")

    return df

def get_rsi_daily(ticker, config, time_period=10, api_key=API_KEY):
    # Construct the URL for the API request
    url = f'https://www.alphavantage.co/query?function=RSI&symbol={ticker}&interval=daily&time_period={time_period}&series_type=open&apikey={api_key}'

    try:
        # Send the GET request with a timeout to handle potential network issues
        r = requests.get(url, timeout=10)
        r.raise_for_status()  # Raise an exception for HTTP errors (e.g., 404 or 500)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching RSI data for {ticker}: {e}")
        return None  # Return None to indicate failure

    # Parse the JSON response from the API
    data = r.json()
    
    # Check if 'Technical Analysis: RSI' exists in the response
    if 'Technical Analysis: RSI' not in data:
        print(f"Error: 'Technical Analysis: RSI' not found for {ticker}.")
        return None

    # Extract the RSI data
    rsi_data = data['Technical Analysis: RSI']

    # Convert the data into a DataFrame
    df = pd.DataFrame.from_dict(rsi_data, orient="index")
    
    # Convert the index to datetime format
    df.index = pd.to_datetime(df.index)
    
    # Convert the DataFrame to float type
    df = df.astype(float)
    
    # Rename the index and reset to a regular column
    df = df.rename_axis("Date").reset_index()

    # Ensure the directory exists before saving the file
    output_dir = f"{config['raw_dataset_dir']}/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the DataFrame to a CSV file
    output_file = os.path.join(output_dir, 'RSI.csv')
    df.to_csv(output_file, index=False)  # Set index=False to avoid saving the index as a separate column
    
    print(f"RSI data for {ticker} saved to {output_file}")

    return df

def fetch_data_helper(ticker, config, api_key=API_KEY):
    try:
        get_rsi_daily(ticker, config, api_key=api_key)
        get_macd_daily(ticker,config, api_key=api_key)
        get_ema_daily(ticker, config, api_key=api_key)
        get_sma_daily(ticker, config, api_key=api_key)
        get_fed_fund_daily(ticker,config, api_key=api_key)
        get_crypto_daily(ticker, config, api_key=api_key)
        get_fx_daily(ticker, config, api_key=api_key)
        get_adjusted_daily(ticker, config, output_size='full', api_key=api_key)

    except Exception as e:
        raise RuntimeError(f"Skipping {ticker} due to error: {e}")

def read_csv_safe(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)
    else:
        print(f"Warning: {file_path} does not exist.")
        return None
    
def fetch_data(ticker, config, api_key=API_KEY):
    try:
        # Fetch the data (if it fails, an exception is raised)
        fetch_data_helper(ticker, config, api_key)
    except RuntimeError as e:
        print(e)  # Print the error and return early
        return None  # Skip processing this ticker

    # Paths to your CSV files
    paths = [
        f"{config['raw_dataset_dir']}/{ticker}/TIME_SERIES_DAILY.csv",
        f"{config['raw_dataset_dir']}/{ticker}/EMA.csv",
        f"{config['raw_dataset_dir']}/{ticker}/EUR_USD_FX_DAILY.csv",
        f"{config['raw_dataset_dir']}/{ticker}/FEDERAL_FUNDS_RATE.csv",
        f"{config['raw_dataset_dir']}/{ticker}/MACD.csv",
        f"{config['raw_dataset_dir']}/{ticker}/RSI.csv",
        f"{config['raw_dataset_dir']}/{ticker}/SMA.csv",
    ]

    # Read all the dataframes
    dfs = [read_csv_safe(path) for path in paths]
    # Remove None entries (files that couldn't be read)
    dfs = [df for df in dfs if df is not None]

    # Merge dataframes on 'Date' column
    if dfs:
        merged_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), dfs)
        
        # Ensure the directory exists before saving the dataset
        output_dir = f"{config['processed_dataset_dir']}/{ticker}"
        os.makedirs(output_dir, exist_ok=True)

        # Save the merged DataFrame to CSV
        merged_df.to_csv(f'{output_dir}/dataset.csv', index=False)
        print(f"Dataset saved to {output_dir}/dataset.csv")
        
        # Return the merged DataFrame
        return merged_df
    else:
        print("No valid data available to merge.")
        return None

if __name__ == "__main__":
    config = get_config()
    for ticker in tqdm(config['tickers'], desc="Processing tickers"):
        fetch_data(ticker,config)