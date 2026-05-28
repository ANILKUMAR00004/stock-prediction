import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date,timedelta
from model.predictor import predict_stock
from utils.data_loader import load_stock_data
from utils.indicators import add_indicators

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Stock Prediction System",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: white;
}

.stMetric {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# TITLE

st.title("📈 AI Powered Stock Prediction Dashboard")
st.markdown("### LSTM Deep Learning Based Multi-Stock Forecasting")

st.divider() # Adds a nice horizontal line to separate title from settings

st.subheader("📊 Stock Settings")

stock_options = {
    # US Stocks
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "NVIDIA": "NVDA",

    # Indian Stocks
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Wipro": "WIPRO.NS",
    "SBI": "SBIN.NS",
    "Adani Enterprises": "ADANIENT.NS"
}
col1, col2, col3 = st.columns(3)

with col1:
    selected_stocks = st.selectbox(
        "Search and Select Stocks", 
        options=list(stock_options.keys()),
        index=None,
        placeholder="Select a Stock"
    )

if "start_date" not in st.session_state:
    st.session_state.start_date = None

if "end_date" not in st.session_state:
    st.session_state.end_date = None
if (
    st.session_state.start_date and
    st.session_state.end_date
):

    days_difference = (
        st.session_state.end_date -
        st.session_state.start_date
    ).days

    if days_difference < 120:
        st.warning(
            f"""
            ⚠ Start date automatically adjusted
            to maintain minimum 120 days data range.
            """
        )

        st.session_state.start_date = (
            st.session_state.end_date -
            timedelta(days=120)
        )

# START DATE
with col2:
    start_date = st.date_input(
        "Start Date",
        value=st.session_state.start_date,
        format="DD-MM-YYYY",
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        key="start_date"
    )

# END DATE
with col3:
    end_date = st.date_input(
        "End Date",
        value=st.session_state.end_date,
        format="DD-MM-YYYY",
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        key="end_date"
    )

all_fields_filled = (
    selected_stocks is not None and
    start_date is not None and
    end_date is not None
)
predict_button = st.button("🚀 Predict Stocks",disabled=not all_fields_filled)
st.divider()
if predict_button:

    ticker = stock_options[selected_stocks]
    st.header(f"📈 {selected_stocks}")
    # LOAD DATA
    df = load_stock_data(
        ticker,
        start_date,
        end_date
    )
    if df.empty:
        st.error(f"No data found for {ticker}")

    # ADD INDICATORS
    df = add_indicators(df)

    st.success(f"{ticker} Data Loaded Successfully")

    # CANDLESTICK CHART

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])

    fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # PREDICTION

    try:
        actual, predicted, future_predictions, accuracy = predict_stock(df)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()

    # METRICS

    col1, col2 = st.columns(2)
    col1.metric(
        "Prediction Accuracy",
        f"{accuracy:.2f}%"
    )

    col2.metric(
        "30 Day Forecast Price",
        f"${future_predictions[-1]:.2f}"
    )

    # ACTUAL VS PREDICTED

    prediction_chart = go.Figure()

    prediction_chart.add_trace(go.Scatter(
        y=actual,
        mode='lines',
        name='Actual Price'
    ))
    prediction_chart.add_trace(go.Scatter(
        y=predicted,
        mode='lines',
        name='Predicted Price'
    ))
    prediction_chart.update_layout(
        title=f"{ticker} Actual vs Predicted",
        template="plotly_dark",
        height=500,
        xaxis_title="Time",
        yaxis_title="Price"
    )
    st.plotly_chart(
        prediction_chart,
        use_container_width=True
    )
    # FUTURE FORECAST
    future_chart = go.Figure()
    future_chart.add_trace(go.Scatter(
        y=future_predictions,
        mode='lines+markers',
        name='30 Day Forecast'
    ))
    future_chart.update_layout(
        title=f"{ticker} Future 30-Day Forecast",
        template="plotly_dark",
        height=500,
        xaxis_title="Future Days",
        yaxis_title="Predicted Price"
    )
    st.plotly_chart(
        future_chart,
        use_container_width=True
    )
    # SHOW DATA
    st.subheader("📄 Latest Stock Data")
    st.dataframe(
        df.tail(20),
        use_container_width=True
    )
    st.divider()