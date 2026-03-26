import pandas as pd
import pandas_ta as ta
import numpy as np

def preprocess_data(df):
    """
    High-accuracy preprocessing with a rich set of technical indicators.
    Features: trend, momentum, volatility, volume, pattern, and calendar signals.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.ffill().dropna()

    # ── Price-derived features ──────────────────────────────────────────────
    df['Returns']      = df['Close'].pct_change()
    df['Log_Returns']  = np.log(df['Close'] / df['Close'].shift(1))
    df['HL_Spread']    = (df['High'] - df['Low']) / df['Close']
    df['OC_Spread']    = (df['Close'] - df['Open']) / df['Open']

    # ── Trend indicators ────────────────────────────────────────────────────
    df.ta.ema(length=9,  append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)

    # EMA crossover signals
    df['EMA9_20_cross']  = df['EMA_9'] - df['EMA_20']
    df['EMA20_50_cross'] = df['EMA_20'] - df['EMA_50']
    df['Price_EMA20']    = df['Close'] - df['EMA_20']
    df['Price_EMA50']    = df['Close'] - df['EMA_50']

    # ── Momentum indicators ─────────────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=7,  append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.roc(length=10, append=True)   # Rate of Change
    df.ta.roc(length=20, append=True)
    df.ta.mom(length=10, append=True)   # Momentum
    df.ta.willr(length=14, append=True) # Williams %R
    df.ta.cci(length=20, append=True)   # CCI

    # Stochastic oscillator
    df.ta.stoch(append=True)

    # ── Volatility indicators ───────────────────────────────────────────────
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)   # Average True Range

    df['BB_Width']     = (df.get('BBU_20_2.0', 0) - df.get('BBL_20_2.0', 0)) / df.get('BBM_20_2.0', 1)
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_21']= df['Returns'].rolling(21).std()

    # ── Volume indicators ───────────────────────────────────────────────────
    if 'Volume' in df.columns:
        df.ta.obv(append=True)              # On-Balance Volume
        df.ta.vwap(append=True)             # VWAP
        df['Volume_Ratio']  = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Price_Volume']  = df['Close'] * df['Volume']

    # ── Rolling statistics ──────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f'Roll_Mean_{w}']  = df['Close'].rolling(w).mean()
        df[f'Roll_Std_{w}']   = df['Close'].rolling(w).std()
        df[f'Roll_Max_{w}']   = df['Close'].rolling(w).max()
        df[f'Roll_Min_{w}']   = df['Close'].rolling(w).min()
        df[f'Dist_Max_{w}']   = (df['Close'] - df[f'Roll_Max_{w}']) / df[f'Roll_Max_{w}']
        df[f'Dist_Min_{w}']   = (df['Close'] - df[f'Roll_Min_{w}']) / df[f'Roll_Min_{w}']

    # ── Calendar features ───────────────────────────────────────────────────
    df['DayOfWeek'] = df.index.dayofweek
    df['Month']     = df.index.month
    df['Quarter']   = df.index.quarter

    # ── 5-Day Targets ───────────────────────────────────────────────────────
    for i in range(1, 6):
        df[f'Target_{i}d'] = df['Close'].shift(-i)

    df = df.dropna()
    return df
