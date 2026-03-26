from src.data_loader import fetch_stock_data
from src.preprocess import preprocess_data
from src.model import train_model
import os

def main():
    # Configuration
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    data_path = f"data/raw/{ticker}.csv"
    
    # 1. Fetch Data
    if not os.path.exists(data_path):
        df = fetch_stock_data(ticker, start_date, end_date, data_path)
    else:
        import pandas as pd
        # Try loading and handle potential multi-level headers in existing CSV
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        if df.iloc[0].dtype == object: # Detect if header rows were loaded as data
            df = pd.read_csv(data_path, index_col=0, parse_dates=True, header=[0,1])
            df.columns = df.columns.get_level_values(0)
        print(f"Loaded existing data from {data_path}")
    
    if df is None:
        print("Failed to fetch data. Exiting.")
        return
    
    # 2. Preprocess Data
    print("Preprocessing data...")
    processed_df = preprocess_data(df)
    
    # 3. Train Model
    print("Training model...")
    model, mse = train_model(processed_df)
    
    print("\nProject Initialized Successfully!")
    print(f"Baseline Random Forest MSE for {ticker}: {mse:.4f}")

if __name__ == "__main__":
    main()
