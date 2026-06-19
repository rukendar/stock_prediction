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

    # Normalize EMAs (percentage difference relative to Close)
    df['EMA_9_ratio']   = (df['Close'] - df['EMA_9']) / df['Close']
    df['EMA_20_ratio']  = (df['Close'] - df['EMA_20']) / df['Close']
    df['EMA_50_ratio']  = (df['Close'] - df['EMA_50']) / df['Close']
    df['EMA_200_ratio'] = (df['Close'] - df['EMA_200']) / df['Close']

    # EMA crossover signals (normalized)
    df['EMA9_20_cross']  = (df['EMA_9'] - df['EMA_20']) / df['Close']
    df['EMA20_50_cross'] = (df['EMA_20'] - df['EMA_50']) / df['Close']
    df['Price_EMA20']    = (df['Close'] - df['EMA_20']) / df['Close']
    df['Price_EMA50']    = (df['Close'] - df['EMA_50']) / df['Close']

    # ── Momentum indicators ─────────────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=7,  append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.roc(length=10, append=True)   # Rate of Change
    df.ta.roc(length=20, append=True)
    df.ta.mom(length=10, append=True)   # Momentum
    df.ta.willr(length=14, append=True) # Williams %R
    df.ta.cci(length=20, append=True)   # CCI
    df.ta.adx(length=14, append=True)   # Average Directional Index (Trend Strength)

    # Normalize MACD and Momentum
    if 'MACD_12_26_9' in df.columns:
        df['MACD_12_26_9_ratio']  = df['MACD_12_26_9'] / df['Close']
        df['MACDh_12_26_9_ratio'] = df['MACDh_12_26_9'] / df['Close']
        df['MACDs_12_26_9_ratio'] = df['MACDs_12_26_9'] / df['Close']
    if 'MOM_10' in df.columns:
        df['MOM_10_ratio'] = df['MOM_10'] / df['Close']

    # Stochastic oscillator
    df.ta.stoch(append=True)

    # ── Volatility indicators ───────────────────────────────────────────────
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)   # Average True Range

    # Normalize Bollinger Bands and ATR
    bbu_col = next((c for c in df.columns if c.startswith('BBU_20')), None)
    bbl_col = next((c for c in df.columns if c.startswith('BBL_20')), None)
    bbm_col = next((c for c in df.columns if c.startswith('BBM_20')), None)
    atr_col = next((c for c in df.columns if c.startswith('ATR_14') or c.startswith('ATRr_14')), None)

    # Fallback default names
    fallback_u = bbu_col or ('BBU_20_2.0_2.0' if 'BBU_20_2.0_2.0' in df.columns else 'BBU_20_2.0')
    fallback_l = bbl_col or ('BBL_20_2.0_2.0' if 'BBL_20_2.0_2.0' in df.columns else 'BBL_20_2.0')
    fallback_m = bbm_col or ('BBM_20_2.0_2.0' if 'BBM_20_2.0_2.0' in df.columns else 'BBM_20_2.0')
    fallback_atr = atr_col or ('ATRr_14' if 'ATRr_14' in df.columns else 'ATR_14')

    df['BBU_20_2.0_ratio'] = (df.get(fallback_u, df['Close']) - df['Close']) / df['Close']
    df['BBL_20_2.0_ratio'] = (df['Close'] - df.get(fallback_l, df['Close'])) / df['Close']
    df['BBM_20_2.0_ratio'] = (df['Close'] - df.get(fallback_m, df['Close'])) / df['Close']
    df['NATR_14']          = df.get(fallback_atr, 0.0) / df['Close']

    df['BB_Width']     = (df.get(fallback_u, 0) - df.get(fallback_l, 0)) / df.get(fallback_m, 1)
    df['Volatility_5'] = df['Returns'].rolling(5).std()
    df['Volatility_21']= df['Returns'].rolling(21).std()

    # ── Volume indicators ───────────────────────────────────────────────────
    if 'Volume' in df.columns:
        df.ta.obv(append=True)              # On-Balance Volume
        df.ta.vwap(append=True)             # VWAP
        df['Volume_Ratio']  = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Price_Volume']  = df['Close'] * df['Volume']
        
        # Normalize VWAP and OBV
        df['VWAP_ratio'] = (df['Close'] - df['VWAP_D']) / df['Close']
        df['OBV_ratio']  = df['OBV'] / df['Volume'].rolling(20).mean()

    # ── Rolling statistics ──────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f'Roll_Mean_{w}']        = df['Close'].rolling(w).mean()
        df[f'Roll_Std_{w}']         = df['Close'].rolling(w).std()
        df[f'Roll_Max_{w}']         = df['Close'].rolling(w).max()
        df[f'Roll_Min_{w}']         = df['Close'].rolling(w).min()
        df[f'Roll_Mean_{w}_ratio']  = (df['Close'] - df[f'Roll_Mean_{w}']) / df['Close']
        df[f'Roll_Std_{w}_ratio']   = df[f'Roll_Std_{w}'] / df['Close']
        df[f'Dist_Max_{w}']         = (df['Close'] - df[f'Roll_Max_{w}']) / df[f'Roll_Max_{w}']
        df[f'Dist_Min_{w}']         = (df['Close'] - df[f'Roll_Min_{w}']) / df[f'Roll_Min_{w}']

    # ── Calendar features ───────────────────────────────────────────────────
    df['DayOfWeek'] = df.index.dayofweek
    df['Month']     = df.index.month
    df['Quarter']   = df.index.quarter

    # ── Sequential Lag Features (CRITICAL for ML strictly viewing history) ──
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
        df[f'Vol_Lag_{lag}'] = df['Volume'].shift(lag) if 'Volume' in df.columns else 0
        
        # Normalize Lags
        df[f'Close_Lag_ratio_{lag}'] = (df['Close'] - df[f'Close_Lag_{lag}']) / df['Close']
        if 'Volume' in df.columns:
            df[f'Vol_Lag_Ratio_{lag}'] = df[f'Vol_Lag_{lag}'] / df['Volume'].rolling(20).mean()
        else:
            df[f'Vol_Lag_Ratio_{lag}'] = 0

    # ── 5-Day Targets ───────────────────────────────────────────────────────
    for i in range(1, 6):
        df[f'Target_{i}d'] = (df['Close'].shift(-i) - df['Close']) / df['Close']

    # Only drop rows where FEATURES are missing (e.g., from historical lags).
    # We must NEVER drop rows where TARGETS are missing, otherwise we delete the last 5 days of live market data!
    feature_columns = [c for c in df.columns if not c.startswith('Target_')]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns)
    
    # ── Multicollinearity Pruning ───────────────────────────────────────────
    # Drop features that have correlation > 0.95 with other features to reduce model variance
    collinear_features = [
        'Log_Returns', 'Price_EMA20', 'Price_EMA50', 'ADXR_14_2', 
        'MACD_12_26_9_ratio', 'STOCHd_14_3_3', 'BBP_20_2.0_2.0', 
        'BBM_20_2.0_ratio', 'BB_Width', 'Roll_Mean_10_ratio', 
        'Roll_Mean_20_ratio', 'Roll_Std_20_ratio', 'Roll_Mean_50_ratio', 
        'Quarter'
    ]
    df = df.drop(columns=[col for col in collinear_features if col in df.columns], errors='ignore')
    
    return df
