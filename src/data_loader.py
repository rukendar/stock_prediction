import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker, start_date=None, end_date=None, period="max", save_directory="data/raw"):
    """
    Fetches historical stock data for a given ticker or list of tickers.
    Enhanced with retry logic for different periods if 'max' fails.
    """
    os.makedirs(save_directory, exist_ok=True)
    
    if isinstance(ticker, list):
        print(f"Fetching batch data for {len(ticker)} stocks with period={period}...")
        try:
            if start_date and end_date:
                data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker')
            else:
                data = yf.download(ticker, period=period, group_by='ticker')
            
            for t in ticker:
                try:
                    if len(ticker) > 1:
                        if t in data:
                            ticker_data = data[t].dropna(how='all')
                        else:
                            continue
                    else:
                        ticker_data = data
                    
                    if ticker_data.empty:
                        continue
                    
                    if isinstance(ticker_data.columns, pd.MultiIndex):
                        ticker_data.columns = ticker_data.columns.get_level_values(0)
                    
                    ticker_data = ticker_data[ticker_data.index.notnull()]
                    save_path = os.path.join(save_directory, f"{t}.csv")
                    ticker_data.to_csv(save_path)
                except Exception as e:
                    print(f"Error saving {t}: {e}")
            return data
        except Exception as e:
            print(f"Batch fetch failed: {e}")
            return None
    else:
        # Single ticker logic with robust retry
        periods_to_try = [period, "10y", "5y", "2y", "1y"] if period == "max" else [period]
        
        for p in periods_to_try:
            print(f"Fetching data for {ticker} with period={p}...")
            try:
                if start_date and end_date:
                    data = yf.download(ticker, start=start_date, end=end_date)
                else:
                    data = yf.download(ticker, period=p)
                    
                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    save_path = os.path.join(save_directory, f"{ticker}.csv")
                    data.to_csv(save_path)
                    print(f"Successfully fetched {ticker} with period={p}")
                    return data
            except Exception as e:
                print(f"Failed to fetch {ticker} with period={p}: {e}")
        
        print(f"Error: All fetch attempts for {ticker} failed.")
        return None

if __name__ == "__main__":
    # Test fetch
    fetch_stock_data("AAPL", "2023-01-01", "2023-12-31", "data/raw/AAPL.csv")
