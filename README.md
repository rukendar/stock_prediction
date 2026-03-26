# Stock Price Prediction Project

A machine learning project to predict stock prices using historical data.

## Project Structure
- `data/`: Raw and processed stock data.
- `notebooks/`: EDA and model experimentation.
- `src/`: Source code for the project.
  - `data_loader.py`: Script to fetch data.
  - `preprocess.py`: Data cleaning and feature engineering.
  - `model.py`: ML model training and evaluation.
- `main.py`: Entry point for the application.
- `requirements.txt`: Project dependencies.

## Advanced Features
- **Nifty 100 Coverage**: AI models for the top 100 Indian stocks.
- **5-Day Forecasting**: High-precision XGBoost models for multi-day outlooks.
- **Premium Dashboard**: Groww-style interactive candlestick charts and metrics.
- **Technical Indicators**: RSI, EMA, MACD, and Bollinger Bands integrated into the AI logic.

## Setup & Usage
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train Models** (Collects data and trains 500+ models):
   ```bash
   python train_all.py
   ```
3. **Launch Dashboard**:
   ```bash
   streamlit run app.py
   ```

## Project structure
- `data/raw/`: Historical CSV data for all 100 stocks.
- `models/`: Serialized XGBoost models (`.joblib`).
- `src/`: Core logic (Data loading, Preprocessing, Modeling, Stock lists).
- `app.py`: Streamlit User Interface.
