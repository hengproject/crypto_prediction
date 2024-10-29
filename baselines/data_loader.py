import pandas as pd
from datetime import datetime

# Define the data loader function
def load_crypto_data(file_path):
    df = pd.read_csv(file_path)

    # Convert timestamp to datetime for better readability
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    # Convert the 'is_closed' column to boolean if it's not already
    df['is_closed'] = df['is_closed'].astype(bool)

    # Ensure correct data types for numeric columns
    numeric_columns = ['vol_as_u', 'close', 'high', 'low', 'open']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Drop rows with NaN values (optional, depending on your model's needs)
    df.dropna(inplace=True)

    # Filter only rows with the type 'future' (if relevant)
    df = df[df['type'] == 'future']

    df.set_index('timestamp', inplace=True)

    # Resample to 1-minute intervals, aggregating as needed
    minute_df = df.resample('1min').agg({
        'vol_as_u': 'sum',  # Sum volume within the minute
        'open': 'first',  # First open price in the minute
        'close': 'last',  # Last close price in the minute
        'high': 'max',  # Highest price within the minute
        'low': 'min',  # Lowest price within the minute
        'is_closed': 'last',  # Whether the minute closed (should generally be True)
        'interval': 'min',  # Keep the original interval type for reference
        'type': 'last',  # Keep the type for reference
        'pair': 'last'  # Keep the trading pair for reference
    }).dropna()  # Drop rows with missing data after resampling

    minute_df['interval'] = '1min'

    return minute_df


# Example usage
file_path = '../btc_future_only_10s.csv'
crypto_data = load_crypto_data(file_path)
print(crypto_data.head())
