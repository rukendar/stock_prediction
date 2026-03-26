from src.data_loader import fetch_stock_data
from src.preprocess import preprocess_data
from src.model import train_stock_model
import pandas as pd
import os

INDICES = ["^BSESN", "^NSEI", "^NSMIDCP"]

def train_indices():
    for ticker in INDICES:
        data_path = f"data/raw/{ticker}.csv"
        print(f"Fetching data for {ticker}...")
        fetch_stock_data(ticker, period="10y") # 10 yrs is plenty for indices
        
        if os.path.exists(data_path):
            print(f"Training model for {ticker}...")
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            if df.empty:
                print(f"Warning: {ticker} CSV is empty.")
                continue
            
            processed_df = preprocess_data(df)
            train_stock_model(processed_df, ticker)

if __name__ == "__main__":
    train_indices()
