from src.stock_list import NIFTY_100_STOCKS
from src.data_loader import fetch_stock_data
from src.preprocess import preprocess_data
from src.model import train_stock_model
import pandas as pd
import os

def main():
    # 1. Process each stock: Fetch (if missing) then Train
    print(f"Processing {len(NIFTY_100_STOCKS)} stocks...")
    
    for ticker in NIFTY_100_STOCKS:
        data_path = f"data/raw/{ticker}.csv"
        
        # Fetch if missing
        if not os.path.exists(data_path):
            print(f"Fetching missing data for {ticker}...")
            fetch_stock_data(ticker, period="max")
            
        # Train if data exists
        if os.path.exists(data_path):
            try:
                print(f"Training model for {ticker}...")
                df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                if df.empty:
                    print(f"Warning: {ticker} CSV is empty.")
                    continue
                
                processed_df = preprocess_data(df)
                train_stock_model(processed_df, ticker)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
        else:
            print(f"Critical: Could not obtain data for {ticker}")

if __name__ == "__main__":
    main()
