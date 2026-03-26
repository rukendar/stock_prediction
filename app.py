import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.stock_list import NIFTY_100_STOCKS
from src.preprocess import preprocess_data
from src.model import predict_future
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Indian Stock AI Predictor", layout="wide", page_icon="📈")

# Premium CSS for Groww-style UI
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .stMetric {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
        transition: all 0.2s ease;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
    }
    .stTable {
        background-color: #161b22;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Indian Stock Market Prediction Dashboard")
st.markdown("### Professional AI-Powered Forecasts (Nifty 100)")
st.sidebar.markdown("## 📊 Stock Explorer")

# Sidebar for selection
ticker = st.sidebar.selectbox("Select Investment", NIFTY_100_STOCKS)

# Load data
data_path = f"data/raw/{ticker}.csv"

if os.path.exists(data_path):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    
    # Header Statistics
    curr_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change = curr_price - prev_price
    change_pct = (change / prev_price) * 100
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Current Price", f"₹{curr_price:,.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    m_col2.metric("Day High", f"₹{df['High'].iloc[-1]:,.2f}")
    m_col3.metric("Day Low", f"₹{df['Low'].iloc[-1]:,.2f}")
    m_col4.metric("Volume", f"{df['Volume'].iloc[-1]:,}")

    # Display Chart
    st.subheader("🔍 Technical Analysis")
    
    fig = go.Figure(data=[go.Candlestick(x=df.index[-90:],
                open=df['Open'][-90:],
                high=df['High'][-90:],
                low=df['Low'][-90:],
                close=df['Close'][-90:],
                name='Price Action')])
    
    # Add Moving Averages
    df['EMA20'] = df['Close'].ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=df.index[-90:], y=df['EMA20'][-90:], name='EMA 20', line=dict(color='#ff7f0e', width=1)))

    fig.update_layout(
        title=f"{ticker} Candlestick Chart (Active View)",
        yaxis_title="Price (INR)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions
    st.markdown("---")
    st.subheader("🔮 5-Day Precision Outlook")
    
    processed_df = preprocess_data(df)
    predictions = predict_future(processed_df, ticker)
    
    cols = st.columns(5)
    dates = [datetime.now() + timedelta(days=i) for i in range(1, 6)]
    
    for i, col in enumerate(cols):
        day_name = dates[i].strftime("%A")
        date_str = dates[i].strftime("%d %b")
        pred_val = predictions[i]
        if pred_val:
            delta = pred_val - curr_price
            delta_pct = (delta / curr_price) * 100
            col.metric(f"{day_name}", f"₹{pred_val:,.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
        else:
            col.metric(f"{day_name}", "TBC", "Model training...")

    # Past 5-Day Comparison
    st.markdown("---")
    st.subheader("📑 Historical Prediction Validation")
    
    past_df = processed_df.tail(6).copy()
    if len(past_df) >= 6:
        actuals = past_df['Close'].values[-5:]
        dates_past = past_df.index.strftime("%d %b %Y")[-5:]
        
        # Use 1D target from previous day as "Predicted" for the actual day
        predicteds = past_df['Target_1d'].shift(1).dropna().values[-5:]
        
        comp_df = pd.DataFrame({
            "Date": dates_past,
            "Actual Price (₹)": [f"{x:,.2f}" for x in actuals],
            "AI Predicted (₹)": [f"{x:,.2f}" for x in predicteds],
            "Accuracy Variance": [f"{abs(a-p)/a*100:.2f}%" for a, p in zip(actuals, predicteds)]
        })
        st.table(comp_df)
    else:
        st.info("Additional data points needed for accuracy validation.")

else:
    st.warning(f"🚧 Data for **{ticker}** is still being initialized. Please run the training script.")

st.sidebar.markdown("---")
st.sidebar.success("✅ Model: XGBoost Regression")
st.sidebar.info("Accuracy is improved using Momentum & Volatility indicators.")
st.info("Disclaimer: Predictions are based on historical data and AI models. Stock markets involve risks.")
