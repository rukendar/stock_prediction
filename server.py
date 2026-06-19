from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from models import db, User
import os
import pandas as pd
import threading
import copy
from src.preprocess import preprocess_data
from src.model import predict_future, get_cached_joblib, NON_STATIONARY_COLS
from src.stock_list import NIFTY_100_STOCKS, STOCK_NAMES, STOCK_DOMAINS, INDEX_DOMAINS
from src.sentiment import fetch_live_sentiment
import jwt
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'growup-secret-key-12345')

# Cloud PostgreSQL URI mapping with SQLite fallback
db_url = os.environ.get('DATABASE_URL', 'sqlite:///growup.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
db.init_app(app)

# Initialize Database
with app.app_context():
    db.create_all()

# Cache threading locks
_index_cache_lock = threading.Lock()
_ohlc_cache_lock = threading.Lock()
_intraday_cache_lock = threading.Lock()
_predict_cache_lock = threading.Lock()
_last_update_check_lock = threading.Lock()

# --- Auth Middleware ---
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get('token')
        if not token:
            return redirect(url_for('login_page'))
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = db.session.get(User, data['user_id'])
            if not current_user:
                return redirect(url_for('login_page'))
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
    return render_template('dashboard.html', user=current_user, stocks=NIFTY_100_STOCKS, stock_names=STOCK_NAMES, stock_domains=STOCK_DOMAINS, indices=INDICES, index_domains=INDEX_DOMAINS)

# --- API Endpoints ---

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    if not data or not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({"message": "Missing required fields"}), 400
        
    if User.query.filter_by(username=data['username']).first():
        return jsonify({"message": "Username already exists"}), 400
        
    if User.query.filter_by(email=data['email']).first():
        return jsonify({"message": "Email already registered"}), 400
    
    new_user = User(username=data['username'], email=data['email'])
    new_user.set_password(data['password'])
    db.session.add(new_user)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Database error during signup"}), 500
        
    return jsonify({"message": "User created successfully"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not all(k in data for k in ('username', 'password')):
        return jsonify({"message": "Missing username or password"}), 400
        
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

_index_cache = {}
_ohlc_cache = {}
_intraday_cache = {}

@app.route('/api/index/<path:ticker>')
@token_required
def index_data(current_user, ticker):
    """Fetch historical OHLC data for a market index (no ML model)."""
    now = datetime.now()
    with _index_cache_lock:
        cache_entry = _index_cache.get(ticker)
    if cache_entry:
        data, expiry = cache_entry
        if now < expiry:
            return jsonify(data)
            
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
        res_data = {
            "ticker": ticker,
            "company_name": INDICES.get(ticker, ticker),
            "current_price": current_price,
            "history": history,
            "is_index": True
        }
        # Cache for 15 minutes (900 seconds)
        with _index_cache_lock:
            _index_cache[ticker] = (res_data, now + timedelta(seconds=900))
        return jsonify(res_data)
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
    period   = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    cache_key = f"{ticker}_{period}_{interval}"
    
    now = datetime.now()
    with _ohlc_cache_lock:
        cache_entry = _ohlc_cache.get(cache_key)
    if cache_entry:
        data, expiry = cache_entry
        if now < expiry:
            return jsonify(data)
            
    try:
        import yfinance as yf
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
        res_data = {"candles": candles, "interval": interval, "date_key": date_col}
        # Cache for 5 minutes (300 seconds)
        with _ohlc_cache_lock:
            _ohlc_cache[cache_key] = (res_data, now + timedelta(seconds=300))
        return jsonify(res_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/intraday/<ticker>')
@token_required
def intraday(current_user, ticker):
    """Fetches live 5-minute OHLC candles for today's trading session."""
    now = datetime.now()
    with _intraday_cache_lock:
        cache_entry = _intraday_cache.get(ticker)
    if cache_entry:
        data, expiry = cache_entry
        if now < expiry:
            return jsonify(data)
            
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
        res_data = {
            "ticker": ticker,
            "interval": "5m",
            "candles": records
        }
        # Cache for 60 seconds
        with _intraday_cache_lock:
            _intraday_cache[ticker] = (res_data, now + timedelta(seconds=60))
        return jsonify(res_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

_last_update_check = {}
_predict_cache = {}

def log_model_monitoring(ticker, validation):
    import numpy as np
    import os
    os.makedirs("data", exist_ok=True)
    log_path = "data/model_monitoring.log"
    
    if not validation:
        return
        
    pct_errors = [v['variance_pct'] for v in validation]
    if len(pct_errors) >= 3:
        variance_pct = float(np.var(pct_errors))
        mean_pct_error = float(np.mean(np.abs(pct_errors)))
        
        log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ticker: {ticker} | 5-day Mean Abs Error %: {mean_pct_error:.2f}% | 5-day Error Variance: {variance_pct:.2f}%"
        
        if variance_pct > 5.0:
            log_entry += " | WARNING: 5-day rolling prediction variance exceeds 5% threshold!"
            
        try:
            with open(log_path, "a") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"Failed to write to monitoring log: {e}")

@app.route('/api/predict/<ticker>')
@token_required
def predict(current_user, ticker):
    global _last_update_check, _predict_cache
    import time
    t0 = time.time()
    
    now = datetime.now()
    with _predict_cache_lock:
        cache_entry = _predict_cache.get(ticker)
        
    if cache_entry:
        cached_res_raw, expiry = cache_entry
        if now < expiry:
            # Mutate a DEEP COPY of cached_res, not the shared cached dict reference!
            cached_res = copy.deepcopy(cached_res_raw)
            # Update the live price dynamically inside the cached response!
            from src.data_loader import get_live_share_price
            new_live_price = get_live_share_price(ticker)
            cached_res["current_price"] = new_live_price
            
            # Predictions remain anchored to hist_base_price (calculated at cache miss time)
            # to remain look-ahead free, so we DO NOT dynamically scale them by live intraday price!
            return jsonify(cached_res)
    
    data_path = f"data/raw/{ticker}.csv"
    if not os.path.exists(data_path):
        return jsonify({"error": "Data not found"}), 404
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    t1 = time.time()
    
    from src.data_loader import get_live_share_price
    current_price, actual_prev_close = get_live_share_price(ticker, return_prev_close=True)
    
    # Sync today's active candle in the historical dataframe with the ultra-live price 
    # so the Validation table "Actual" matches the Dashboard's live price perfectly.
    if len(df) > 0 and df.index[-1].date() == now.date() and current_price > 0:
        if 'Close' in df.columns:
            df.iloc[-1, df.columns.get_loc('Close')] = current_price
    
    with _last_update_check_lock:
        last_check = _last_update_check.get(ticker, datetime.min)
    
    # Check if the last data point is prior to today. If so, and we haven't asked Yahoo Finance in the last hour, fetch in background!
    if df.index[-1].date() < now.date() and (now - last_check).total_seconds() > 3600:
        with _last_update_check_lock:
            _last_update_check[ticker] = now
        import threading
        from src.data_loader import update_daily_data
        def run_update():
            try:
                update_daily_data(ticker)
            except Exception as e:
                print(f"Background update failed for {ticker}: {e}")
        threading.Thread(target=run_update).start()
    t2 = time.time()
        
    # Preprocess the entire historical dataframe to prevent technical indicator convergence loss
    processed_df = preprocess_data(df)
    # Take tail 150 rows for feature subsetting to keep prediction/validation steps fast
    processed_df_subset = processed_df.tail(150) if len(processed_df) > 150 else processed_df
    t3 = time.time()
    
    # Predict next 5 days
    predictions_technical = predict_future(processed_df_subset, ticker)
    t4 = time.time()
    
    # NLP Sentiment Module Injection
    sentiment_data = {"compound": 0.0, "multiplier": 1.0, "dynamic_slope": 0.02, "label": "No Recent News", "articles": []}
    multiplier = 1.0
    try:
        sentiment_data = fetch_live_sentiment(ticker)
        multiplier = sentiment_data.get('multiplier', 1.0)
    except Exception as e:
        print(f"Failed NLP Parsing: {e}")
    t5 = time.time()
        
    predictions_sentiment = []
    for idx, p in enumerate(predictions_technical):
        if p is not None:
            # Implement Human-Like Market Psychology: News Sentiment Decay
            # News shocks have max impact on Day 1, and fade back to technical fundamentals by Day 5.
            sentiment_impact = multiplier - 1.0
            decay_factor = max(0.0, 1.0 - (idx * 0.2))  # Day 1: 100%, Day 5: 20%
            decayed_multiplier = 1.0 + (sentiment_impact * decay_factor)
            predictions_sentiment.append(p * decayed_multiplier)
        else:
            predictions_sentiment.append(None)
    
    # Limit chart history payload to last 1250 trading days (5 years) to speed up JSON serialization and render times
    chart_data = df.tail(1250).reset_index()
    chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
    t6 = time.time()
    
    # --- TRUE Backtest: last 5 stock-market trading days ---
    validation = []
    feature_cols = [c for c in processed_df_subset.columns if 'Target' not in c and c not in NON_STATIONARY_COLS]
    
    backtest_df = processed_df_subset.tail(6)   # grab 6 rows: index 0 is base day, 1-5 are the 5 actual future days
    if len(backtest_df) >= 2:
        base_row_df = backtest_df.iloc[[0]]
        base_price = float(base_row_df['Close'].iloc[0])
        rows_available = len(backtest_df)
        
        for i in range(1, rows_available):
            model_path  = os.path.join("models", f"{ticker}_day{i}.joblib")
            scaler_path = os.path.join("models", f"{ticker}_scaler{i}.joblib")
            
            model = get_cached_joblib(model_path)
            if model is not None:
                scaler = get_cached_joblib(scaler_path)
                is_legacy = False
                
                if scaler is not None and hasattr(scaler, 'feature_names_in_'):
                    expected_cols = list(scaler.feature_names_in_)
                    safe_row = base_row_df.copy()
                    for col in expected_cols:
                        if col not in safe_row.columns:
                            safe_row[col] = 0.0
                    feature_row = scaler.transform(safe_row[expected_cols])
                    is_legacy = 'Close' in expected_cols
                elif scaler is not None:
                    feature_row = scaler.transform(base_row_df[feature_cols].values)
                else:
                    feature_row = base_row_df[feature_cols].values
                    is_legacy = True
 
                actual_date   = backtest_df.index[i].strftime('%d %b %Y')
                actual_price  = float(backtest_df['Close'].iloc[i])
                pred = float(model.predict(feature_row)[0])
                
                if is_legacy:
                    predicted_price = pred
                else:
                    predicted_price = base_price * (1 + pred)
 
                variance_pct  = (predicted_price - actual_price) / actual_price * 100
                variance_abs  = abs(actual_price - predicted_price)
                
                # Prediction is considered correct if the error is less than 0.5%
                direction_correct = abs(variance_pct) < 0.5
 
                validation.append({
                    "date":              actual_date,
                    "actual":            round(actual_price, 2),
                    "predicted":         round(predicted_price, 2),
                    "variance_pct":      round(variance_pct, 2),
                    "variance_abs":      round(variance_abs, 2),
                    "direction_correct": direction_correct,
                    "prev_close":        round(base_price, 2)
                })
    t7 = time.time()
    t8 = time.time()
    
    print(f"PROFILE [{ticker}]: total={t8-t0:.3f}s | read_csv={t1-t0:.3f}s | bg_update={t2-t1:.3f}s | preprocess={t3-t2:.3f}s | predict_future={t4-t3:.3f}s | sentiment={t5-t4:.3f}s | chart_data={t6-t5:.3f}s | validation={t7-t6:.3f}s", flush=True)

    # Use actual authoritative prev_close for the baseline, so 1-day % changes match the exchanges
    hist_base_price = actual_prev_close if actual_prev_close > 0 else float(df['Close'].iloc[-1]) if len(df) > 0 else current_price

    predictions_technical_returns = []
    for p in predictions_technical:
        if p is not None and hist_base_price > 0:
            predictions_technical_returns.append((p - hist_base_price) / hist_base_price)
        else:
            predictions_technical_returns.append(None)
            
    predictions_sentiment_returns = []
    for p in predictions_sentiment:
        if p is not None and hist_base_price > 0:
            predictions_sentiment_returns.append((p - hist_base_price) / hist_base_price)
        else:
            predictions_sentiment_returns.append(None)
 
    predictions_technical_absolute = []
    for r in predictions_technical_returns:
        if r is not None and hist_base_price > 0:
            predictions_technical_absolute.append(hist_base_price * (1 + r))
        else:
            predictions_technical_absolute.append(None)
            
    predictions_sentiment_absolute = []
    for r in predictions_sentiment_returns:
        if r is not None and hist_base_price > 0:
            predictions_sentiment_absolute.append(hist_base_price * (1 + r))
        else:
            predictions_sentiment_absolute.append(None)
 
    # Log model drift tracking
    log_model_monitoring(ticker, validation)

    res_data = {
        "ticker": ticker,
        "company_name": STOCK_NAMES.get(ticker, INDICES.get(ticker, ticker)),
        "domain": STOCK_DOMAINS.get(ticker, INDEX_DOMAINS.get(ticker, "")),
        "current_price": current_price,
        "prev_close": round(hist_base_price, 2),
        "predictions": predictions_sentiment_absolute,
        "predictions_technical": predictions_technical_absolute,
        "predictions_sentiment": predictions_sentiment_absolute,
        "predictions_technical_returns": predictions_technical_returns,
        "predictions_sentiment_returns": predictions_sentiment_returns,
        "validation": validation,
        "history": chart_data[['Date', 'Open', 'High', 'Low', 'Close']].to_dict(orient='records'),
        "sentiment": sentiment_data
    }
    
    # Cache for 5 minutes (300 seconds)
    with _predict_cache_lock:
        _predict_cache[ticker] = (res_data, now + timedelta(seconds=300))
    return jsonify(res_data)

@app.route('/api/sq-hive-status')
@token_required
def sq_hive_status(current_user):
    """Returns whether the SQ Hive API key is configured."""
    from src.sentiment import SQ_HIVE_API_KEY
    return jsonify({"enabled": bool(SQ_HIVE_API_KEY)})

@app.route('/api/sq-hive-news/<path:ticker>')
@token_required
def sq_hive_news(current_user, ticker):
    """
    Direct ScoutQuest SQ Hive news fetch (independent of the predict cache).
    Allows the frontend to refresh news without re-running the ML pipeline.
    Attribution: Powered by ScoutQuest.in
    """
    try:
        from src.sentiment import fetch_live_sentiment
        # Bypass cache for manual user refresh requests
        data = fetch_live_sentiment(ticker, cache_duration=600, bypass_cache=True)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live/<ticker>')
@token_required
def live_price(current_user, ticker):
    """Extremely fast endpoint to get the current real-time market price with caching."""
    try:
        from src.data_loader import get_live_share_price
        price = get_live_share_price(ticker)
        return jsonify({
            "ticker": ticker,
            "live_price": price
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
