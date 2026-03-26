from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from models import db, User
import os
import pandas as pd
from src.preprocess import preprocess_data
from src.model import predict_future
from src.stock_list import NIFTY_100_STOCKS, STOCK_NAMES
import jwt
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = 'growup-secret-key-12345'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///growup.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db.init_app(app)

# Initialize Database
with app.app_context():
    db.create_all()

# --- Auth Middleware ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get('token')
        if not token:
            return redirect(url_for('login_page'))
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
        except:
            return redirect(url_for('login_page'))
        return f(current_user, *args, **kwargs)
    return decorated

# --- Routes ---

@app.route('/')
def index():
    if 'token' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

# Market Indices
INDICES = {
    "^BSESN": "BSE Sensex",
    "^NSEI":  "Nifty 50",
    "^NSMIDCP": "Nifty Midcap 100",
}

@app.route('/dashboard')
@token_required
def dashboard(current_user):
    return render_template('dashboard.html', user=current_user, stocks=NIFTY_100_STOCKS, stock_names=STOCK_NAMES, indices=INDICES)

# --- API Endpoints ---

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"message": "Username already exists"}), 400
    
    new_user = User(username=data['username'], email=data['email'])
    new_user.set_password(data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User created successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data['username']).first()
    if user and user.check_password(data['password']):
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, app.config['SECRET_KEY'])
        session['token'] = token
        return jsonify({"message": "Login successful", "token": token}), 200
    return jsonify({"message": "Invalid credentials"}), 401

@app.route('/api/logout')
def logout():
    session.pop('token', None)
    return redirect(url_for('login_page'))

@app.route('/api/index/<path:ticker>')
@token_required
def index_data(current_user, ticker):
    """Fetch historical OHLC data for a market index (no ML model)."""
    try:
        import yfinance as yf
        # Fetch 5-year history at daily interval
        raw = yf.download(ticker, period="5y", interval="1d", progress=False)
        if raw.empty:
            return jsonify({"error": "No data available for this index"}), 404
        if hasattr(raw.columns, 'levels'):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw['Date'] = pd.to_datetime(raw['Date']).dt.strftime('%Y-%m-%d')
        history = raw[['Date', 'Open', 'High', 'Low', 'Close']].to_dict(orient='records')
        current_price = float(raw['Close'].iloc[-1])
        return jsonify({
            "ticker": ticker,
            "company_name": INDICES.get(ticker, ticker),
            "current_price": current_price,
            "history": history,
            "is_index": True
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ohlc/<path:ticker>')
@token_required
def ohlc_interval(current_user, ticker):
    """
    Returns OHLC candles for a given period and candlestick interval.
    Used by the frontend for Groww-style per-timeframe candlestick granularity.
    Query params: period (e.g. '7d'), interval (e.g. '60m')
    """
    try:
        import yfinance as yf
        period   = request.args.get('period', '1mo')
        interval = request.args.get('interval', '1d')
        
        raw = yf.download(ticker, period=period, interval=interval, progress=False)
        if raw.empty:
            return jsonify({"error": "No data"}), 404
        
        if hasattr(raw.columns, 'levels'):
            raw.columns = raw.columns.get_level_values(0)
        
        raw = raw.reset_index()
        date_col = 'Datetime' if 'Datetime' in raw.columns else 'Date'
        raw[date_col] = pd.to_datetime(raw[date_col]).dt.strftime(
            '%Y-%m-%d %H:%M' if 'Datetime' in raw.columns else '%Y-%m-%d'
        )
        
        candles = raw[[date_col, 'Open', 'High', 'Low', 'Close']].rename(columns={date_col: 'Date'}).to_dict(orient='records')
        return jsonify({"candles": candles, "interval": interval, "date_key": date_col})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/intraday/<ticker>')
@token_required
def intraday(current_user, ticker):
    """Fetches live 5-minute OHLC candles for today's trading session."""
    try:
        import yfinance as yf
        data = yf.download(ticker, period="1d", interval="5m", progress=False)
        
        if data.empty:
            return jsonify({"error": "No intraday data available"}), 404
        
        # Flatten MultiIndex columns if present
        if hasattr(data.columns, 'levels'):
            data.columns = data.columns.get_level_values(0)
        
        data = data.reset_index()
        # Convert Datetime to string
        data['Datetime'] = data['Datetime'].dt.strftime('%Y-%m-%d %H:%M')
        
        records = data[['Datetime', 'Open', 'High', 'Low', 'Close']].to_dict(orient='records')
        return jsonify({
            "ticker": ticker,
            "interval": "5m",
            "candles": records
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/<ticker>')
@token_required
def predict(current_user, ticker):
    data_path = f"data/raw/{ticker}.csv"
    if not os.path.exists(data_path):
        return jsonify({"error": "Data not found"}), 404
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    processed_df = preprocess_data(df)
    # Predict next 5 days
    predictions = predict_future(processed_df, ticker)
    
    # Get full OHLC history for time-range filtering on the client
    chart_data = df.reset_index()
    chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
    
    # --- TRUE Backtest: last 5 stock-market trading days ---
    validation = []
    feature_cols = [c for c in processed_df.columns if 'Target' not in c]
    day1_model_path  = os.path.join("models", f"{ticker}_day1.joblib")
    day1_scaler_path = os.path.join("models", f"{ticker}_scaler1.joblib")

    if os.path.exists(day1_model_path):
        import joblib
        day1_model = joblib.load(day1_model_path)
        day1_scaler = joblib.load(day1_scaler_path) if os.path.exists(day1_scaler_path) else None

        # We need 6 rows: row[i-1] → features to predict row[i] (next trading day)
        # processed_df index only has trading days (weekends/holidays already excluded)
        backtest_df = processed_df.tail(6)   # grab 6 to have 5 full pairs
        rows_available = len(backtest_df)

        for i in range(1, rows_available):
            feature_row = backtest_df.iloc[i - 1][feature_cols].values.reshape(1, -1)
            if day1_scaler is not None:
                feature_row = day1_scaler.transform(feature_row)

            actual_date   = backtest_df.index[i].strftime('%d %b %Y')
            actual_price  = float(backtest_df['Close'].iloc[i])
            prev_price    = float(backtest_df['Close'].iloc[i - 1])
            predicted_price = float(day1_model.predict(feature_row)[0])

            variance_pct  = (actual_price - predicted_price) / actual_price * 100
            variance_abs  = abs(actual_price - predicted_price)
            direction_correct = (predicted_price > prev_price) == (actual_price > prev_price)

            validation.append({
                "date":              actual_date,
                "actual":            round(actual_price, 2),
                "predicted":         round(predicted_price, 2),
                "variance_pct":      round(variance_pct, 2),
                "variance_abs":      round(variance_abs, 2),
                "direction_correct": direction_correct,
                "prev_close":        round(prev_price, 2)
            })

    return jsonify({
        "ticker": ticker,
        "company_name": STOCK_NAMES.get(ticker, INDICES.get(ticker, ticker)),
        "current_price": float(df['Close'].iloc[-1]),
        "predictions": predictions,
        "validation": validation,
        "history": chart_data[['Date', 'Open', 'High', 'Low', 'Close']].to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
