import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import datetime as dt_module
import threading
import numpy as np

# Global re-entrant locks per ticker to serialize writes/updates to raw CSV files
_file_write_locks = {}
_locks_lock = threading.Lock()

def get_ticker_lock(ticker):
    with _locks_lock:
        if ticker not in _file_write_locks:
            _file_write_locks[ticker] = threading.RLock()
        return _file_write_locks[ticker]

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
                    
                    # Normalize index timezone to naive
                    ticker_data.index = pd.to_datetime(ticker_data.index).tz_localize(None)
                    
                    # Filter columns to only include required stock OHLCV
                    cols_to_keep = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in ticker_data.columns]
                    ticker_data = ticker_data[cols_to_keep]
                    
                    ticker_data = ticker_data[ticker_data.index.notnull()]
                    save_path = os.path.join(save_directory, f"{t}.csv")
                    with get_ticker_lock(t):
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
                data = None
                try:
                    if start_date and end_date:
                        data = yf.download(ticker, start=start_date, end=end_date)
                    else:
                        data = yf.download(ticker, period=p)
                except Exception as download_err:
                    print(f"yf.download failed for {ticker} (period={p}): {download_err}")

                if data is None or data.empty:
                    print(f"Trying yf.Ticker history fallback for {ticker} (period={p})...")
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        if start_date and end_date:
                            data = ticker_obj.history(start=start_date, end=end_date)
                        else:
                            data = ticker_obj.history(period=p)
                    except Exception as history_err:
                        print(f"yf.Ticker history fallback failed for {ticker} (period={p}): {history_err}")

                if data is not None and not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    
                    # Normalize index timezone to naive
                    data.index = pd.to_datetime(data.index).tz_localize(None)
                    
                    # Filter columns to only include required stock OHLCV
                    cols_to_keep = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in data.columns]
                    data = data[cols_to_keep]
                    
                    data = data[data.index.notnull()]
                    
                    save_path = os.path.join(save_directory, f"{ticker}.csv")
                    with get_ticker_lock(ticker):
                        data.to_csv(save_path)
                    print(f"Successfully fetched {ticker} with period={p}")
                    return data
            except Exception as e:
                print(f"Failed to fetch {ticker} with period={p}: {e}")
        
        print(f"Error: All fetch attempts for {ticker} failed.")
        return None

def update_daily_data(ticker, save_directory="data/raw"):
    """
    Updates the existing CSV file with the most recent daily stock data.
    """
    file_path = os.path.join(save_directory, f"{ticker}.csv")
    
    with get_ticker_lock(ticker):
        if not os.path.exists(file_path):
            print(f"File for {ticker} does not exist. Fetching from scratch...")
            return fetch_stock_data(ticker, save_directory=save_directory)

        try:
            # Load existing data
            df_existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if df_existing.empty:
                return fetch_stock_data(ticker, save_directory=save_directory)
                
            # Standardize existing index timezone to naive
            df_existing.index = pd.to_datetime(df_existing.index).tz_localize(None)
            
            last_date = df_existing.index[-1]
            now_utc = datetime.now(timezone.utc)
            today_ist = now_utc + timedelta(hours=5, minutes=30)
            
            # If the last data point is today or yesterday and it's morning, we might not need an update,
            # but safely fetching the last 15 days handles all weekend/holiday gaps smoothly.
            print(f"Updating {ticker} (Last recorded date: {last_date.date()})...")
            
            new_data = None
            try:
                new_data = yf.download(ticker, period="1mo", progress=False)
            except Exception as download_err:
                print(f"yf.download update failed for {ticker}: {download_err}")
                
            if new_data is None or new_data.empty:
                print(f"Trying yf.Ticker history update fallback for {ticker}...")
                try:
                    new_data = yf.Ticker(ticker).history(period="1mo")
                except Exception as history_err:
                    print(f"yf.Ticker history update failed for {ticker}: {history_err}")
                    
            if new_data is None or new_data.empty:
                print(f"No new data returned for {ticker}.")
                return df_existing
                
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)
                
            # Normalize index timezone to naive
            new_data.index = pd.to_datetime(new_data.index).tz_localize(None)
            
            # Filter columns to only include required stock OHLCV
            cols_to_keep = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in new_data.columns]
            new_data = new_data[cols_to_keep]
            
            new_data = new_data[new_data.index.notnull()]
            
            # Combine and drop duplicates to ensure we only append genuinely new days/update the latest close
            df_combined = pd.concat([df_existing, new_data])
            # Drop duplicates based on the index (Date) keeping the last (most recent fetch)
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined.sort_index(inplace=True)
            
            df_combined.to_csv(file_path)
            print(f"Successfully updated {ticker}.")
            return df_combined

        except Exception as e:
            print(f"Error updating {ticker}: {e}")
            return None

import threading
import time
import numpy as np

_live_price_cache = {}
_cache_lock = threading.Lock()

def is_market_open():
    """
    Checks if the Indian stock market (NSE/BSE) is currently open.
    Trading hours: Monday-Friday, 9:15 AM to 3:30 PM IST (UTC + 5:30)
    """
    now_utc = datetime.now(timezone.utc)
    ist = now_utc + timedelta(hours=5, minutes=30)
    if ist.weekday() >= 5:
        return False
    market_start = ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_start <= ist <= market_end

def get_live_share_price(ticker, cache_duration=30, save_directory="data/raw", return_prev_close=False):
    """
    Retrieves the most accurate real-time stock/index price with 30s/900s caching
    and full fallback to the local CSV database if network fails or market is closed.
    If return_prev_close is True, returns a tuple (live_price, prev_close).
    """
    now = time.time()
    
    # 1. Thread-safe Cache Lookup
    with _cache_lock:
        if ticker in _live_price_cache:
            price, prev_c, expiry = _live_price_cache[ticker]
            if now < expiry:
                return (price, prev_c) if return_prev_close else price

    # 1b. If market is closed, return the last CSV close price immediately (no network hit)
    market_active = is_market_open()
    effective_cache = cache_duration if market_active else 900
    
    prev_close = 0.0
    
    # Pre-load local CSV values as a baseline
    try:
        file_path = os.path.join(save_directory, f"{ticker}.csv")
        if os.path.exists(file_path):
            df_local = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if len(df_local) >= 2:
                prev_close = float(df_local['Close'].iloc[-2]) # if closed, last is today, prev is yesterday
            if len(df_local) >= 1:
                local_last_close = float(df_local['Close'].iloc[-1])
                if prev_close == 0.0:
                    prev_close = local_last_close
    except Exception as csv_err:
        local_last_close = 0.0
        print(f"CSV closed-market read error for {ticker}: {csv_err}")

    if not market_active and 'local_last_close' in locals() and local_last_close > 0:
        with _cache_lock:
            _live_price_cache[ticker] = (local_last_close, prev_close, now + effective_cache)
        return (local_last_close, prev_close) if return_prev_close else local_last_close

    # 2. Fetch live quote from yfinance
    live_price = None
    try:
        tik = yf.Ticker(ticker)
        
        # Always try to grab the exact official previous close
        try:
            val_prev = float(tik.fast_info.previous_close)
            if not np.isnan(val_prev) and val_prev > 0:
                prev_close = val_prev
        except Exception:
            pass

        # Method A: fast_info (extremely fast attribute lookup)
        try:
            val = float(tik.fast_info.last_price)
            if not np.isnan(val) and val > 0:
                live_price = val
        except Exception:
            pass
            
        # Method B: yfinance download (fast fallback download)
        if live_price is None:
            try:
                # Download 1-day history at 1-min interval
                df_live = yf.download(ticker, period="1d", interval="1m", progress=False)
                if not df_live.empty:
                    if hasattr(df_live.columns, 'levels'):
                        df_live.columns = df_live.columns.get_level_values(0)
                    val = float(df_live['Close'].iloc[-1])
                    if not np.isnan(val) and val > 0:
                        live_price = val
            except Exception:
                pass

        # Method C: Ticker info dict lookup
        if live_price is None:
            try:
                info = tik.info
                val = info.get('currentPrice', info.get('regularMarketPrice'))
                if val and not pd.isna(val) and val > 0:
                    live_price = float(val)
                # Try getting regularMarketPreviousClose if we didn't get it from fast_info
                val_prev2 = info.get('regularMarketPreviousClose', info.get('previousClose'))
                if val_prev2 and not pd.isna(val_prev2) and val_prev2 > 0:
                    prev_close = float(val_prev2)
            except Exception:
                pass

    except Exception as e:
        print(f"Network live price fetch failed for {ticker}: {e}")

    # 3. Fallback to Local CSV Database close price
    if live_price is None and 'local_last_close' in locals() and local_last_close > 0:
        live_price = local_last_close
        print(f"Fallback: Loaded local CSV close price for {ticker}: {live_price}")

    # 4. Cache and Return
    if live_price is not None and not np.isnan(live_price) and live_price > 0.0:
        with _cache_lock:
            _live_price_cache[ticker] = (live_price, prev_close, now + effective_cache)
        return (live_price, prev_close) if return_prev_close else live_price
    
    with _cache_lock:
        if ticker in _live_price_cache:
            p, pc, _ = _live_price_cache[ticker]
            return (p, pc) if return_prev_close else p
            
    return (0.0, 0.0) if return_prev_close else 0.0

if __name__ == "__main__":
    # Test fetch
    fetch_stock_data("AAPL", "2023-01-01", "2023-12-31", "data/raw/AAPL.csv")
