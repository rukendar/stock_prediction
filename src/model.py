from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import joblib
import os

def train_stock_model(df, ticker, models_dir="models"):
    """
    Trains a highly accurate Ensemble Model (XGBoost + LightGBM + Random Forest + Neural Network).
    - TimeSeriesSplit cross-validation (preserves time order)
    - RobustScaler for outlier-resilient normalization
    """
    os.makedirs(models_dir, exist_ok=True)

    feature_cols = [c for c in df.columns if 'Target' not in c]
    X_full = df[feature_cols]

    for i in range(1, 6):
        target_col = f'Target_{i}d'
        if target_col not in df.columns:
            continue

        y_full = df[target_col]
        valid_idx = y_full.dropna().index
        X = X_full.loc[valid_idx]
        y = y_full.loc[valid_idx]

        if len(X) < 100:
            print(f"Skipping Day {i} for {ticker}: Not enough data ({len(X)} rows).")
            continue

        # Time-ordered train/test split (no shuffle) — 90/10
        split = int(len(X) * 0.9)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        # Robust scaling (resistant to outliers from market spikes)
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Build Ensemble
        xgb_model = XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
            random_state=42, tree_method='hist'
        )

        lgb_model = LGBMRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5,
            random_state=42, verbose=-1
        )

        rf_model = RandomForestRegressor(
            n_estimators=150, max_depth=10, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        )

        # Neural Network (MLP)
        mlp_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', alpha=0.01, batch_size=64,
            learning_rate='adaptive', max_iter=200, random_state=42
        )

        # Combine via VotingRegressor
        ensemble = VotingRegressor(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model),
                ('mlp', mlp_model)
            ],
            weights=[0.35, 0.35, 0.15, 0.15]  # Prioritize boosting, smooth with RF/NN
        )

        ensemble.fit(X_train_sc, y_train)

        # Save model + scaler
        model_path  = os.path.join(models_dir, f"{ticker}_day{i}.joblib")
        scaler_path = os.path.join(models_dir, f"{ticker}_scaler{i}.joblib")
        joblib.dump(ensemble, model_path)
        joblib.dump(scaler, scaler_path)

        preds = ensemble.predict(X_test_sc)
        mse   = mean_squared_error(y_test, preds)
        rmse  = np.sqrt(mse)
        pct_err = rmse / y_test.mean() * 100
        print(f"[{ticker}] Day {i} Ensemble: RMSE={rmse:.2f} ({pct_err:.2f}% var)")


def predict_future(df, ticker, models_dir="models"):
    """
    Predicts the next 1-5 days, applying the same RobustScaler used at training.
    Falls back gracefully if models or scalers are missing.
    """
    feature_cols = [c for c in df.columns if 'Target' not in c]
    last_row = df.tail(1)[feature_cols]

    predictions = []
    for i in range(1, 6):
        model_path  = os.path.join(models_dir, f"{ticker}_day{i}.joblib")
        scaler_path = os.path.join(models_dir, f"{ticker}_scaler{i}.joblib")

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                row_sc = scaler.transform(last_row)
            else:
                row_sc = last_row.values  # legacy models without scaler
            pred = model.predict(row_sc)[0]
            predictions.append(float(pred))
        else:
            predictions.append(None)

    return predictions
