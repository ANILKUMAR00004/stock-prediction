import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from model.predictor import predict_stock
from utils.data_loader import load_stock_data, get_stock_profile
from utils.indicators import add_indicators
from utils.position_engine import analyze_stock_position, calculate_position_sizing

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Global AI Stock Terminal & Position Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Terminal CSS
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    .stMetric {
        background-color: #131b2e;
        border: 1px solid #1e293b;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-card {
        background: linear-gradient(135deg, #131b2e 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .signal-badge {
        display: inline-block;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 6px 16px;
        border-radius: 8px;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: #131b2e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #1e293b;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Standard Plotly toolbar configuration to prevent overlaps
PLOTLY_CONFIG = {
    'displaylogo': False,
    'scrollZoom': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'toImageButtonOptions': {'format': 'png', 'height': 600, 'width': 1200}
}

# ======================================================
# GLOBAL MARKETS & MUTUAL FUNDS CATALOG
# ======================================================

MARKET_CATALOG = {
    "🇺🇸 US Equities & Tech": {
        "Apple Inc.": "AAPL",
        "Microsoft Corp.": "MSFT",
        "Nvidia Corp.": "NVDA",
        "Alphabet (Google)": "GOOGL",
        "Amazon.com": "AMZN",
        "Meta Platforms": "META",
        "Tesla Inc.": "TSLA",
        "JPMorgan Chase": "JPM",
        "Berkshire Hathaway": "BRK-B"
    },
    "🇮🇳 Indian Equities (NSE)": {
        "Reliance Industries": "RELIANCE.NS",
        "Tata Consultancy Services (TCS)": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Infosys": "INFY.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "State Bank of India (SBI)": "SBIN.NS",
        "Tata Motors": "TATAMOTORS.NS",
        "ITC Limited": "ITC.NS",
        "Bharti Airtel": "BHARTIARTL.NS",
        "Larsen & Toubro": "LT.NS"
    },
    "🏦 US Mutual Funds & Index Funds": {
        "Vanguard 500 Index Fund (VFIAX)": "VFIAX",
        "Vanguard Total Stock Market (VTSAX)": "VTSAX",
        "Fidelity 500 Index Fund (FXAIX)": "FXAIX",
        "Fidelity Contrafund (FCNTX)": "FCNTX",
        "Vanguard Growth Index Fund (VIGAX)": "VIGAX",
        "T. Rowe Price Blue Chip Growth (TRBCX)": "TRBCX",
        "Schwab S&P 500 Index (SWPPX)": "SWPPX"
    },
    "🇮🇳 Indian Mutual Funds & Index ETFs": {
        "Nippon India Nifty 50 BeES ETF": "NIFTYBEES.NS",
        "Nippon India Bank BeES ETF": "BANKBEES.NS",
        "Nippon India IT BeES ETF": "ITBEES.NS",
        "Nippon India Gold BeES ETF": "GOLDBEES.NS",
        "Nippon India Junior BeES (Next 50)": "JUNIORBEES.NS",
        "SBI Nifty 50 ETF": "SETFNIF50.NS",
        "CPSE ETF": "CPSEETF.NS",
        "ICICI Prudential Bharat 22 ETF": "BHARAT22.NS",
        "Motilal Oswal Nasdaq 100 ETF": "MON100.NS",
        "Mirae Asset Nifty Midcap 150 ETF": "MAM150.NS"
    },
    "🌍 Global Indices & Commodities": {
        "S&P 500 Index": "^GSPC",
        "Nasdaq 100 Index": "^NDX",
        "Nifty 50 Index": "^NSEI",
        "Dow Jones Industrial": "^DJI",
        "Gold Futures": "GC=F",
        "Crude Oil Futures": "CL=F"
    },
    "🪙 Crypto Assets": {
        "Bitcoin (USD)": "BTC-USD",
        "Ethereum (USD)": "ETH-USD",
        "Solana (USD)": "SOL-USD"
    }
}

# ======================================================
# SIDEBAR CONTROLS
# ======================================================

with st.sidebar:
    st.title("⚡ Market Terminal")
    st.markdown("Global Equities, Mutual Funds & AI Sizing Engine")
    st.divider()

    st.subheader("🌐 Asset Class & Market Selection")
    
    market_category = st.selectbox(
        "Select Market Category",
        options=list(MARKET_CATALOG.keys()) + ["🔍 Custom Ticker (Any Stock / Fund Worldwide)"],
        index=0
    )

    if market_category == "🔍 Custom Ticker (Any Stock / Fund Worldwide)":
        ticker_input = st.text_input(
            "Enter Ticker Symbol",
            value="AAPL",
            help="Enter any valid ticker from Yahoo Finance (e.g. VFIAX, NIFTYBEES.NS, AMD, TSM, BABA, TATAPOWER.NS)"
        ).strip().upper()
        selected_name = ticker_input
        ticker = ticker_input
    else:
        stocks_in_market = MARKET_CATALOG[market_category]
        selected_name = st.selectbox(
            "Select Fund / Asset",
            options=list(stocks_in_market.keys()),
            index=0
        )
        ticker = stocks_in_market[selected_name]

    st.divider()
    st.subheader("⏱ Timeframe Range")

    timeframe_preset = st.radio(
        "Quick Presets",
        options=["6 Months", "1 Year", "3 Years", "5 Years", "Custom Range"],
        index=2,
        horizontal=True
    )

    today = date.today()
    if timeframe_preset == "6 Months":
        start_date = today - timedelta(days=180)
        end_date = today
    elif timeframe_preset == "1 Year":
        start_date = today - timedelta(days=365)
        end_date = today
    elif timeframe_preset == "3 Years":
        start_date = today - timedelta(days=365 * 3)
        end_date = today
    elif timeframe_preset == "5 Years":
        start_date = today - timedelta(days=365 * 5)
        end_date = today
    else:
        col_s, col_e = st.columns(2)
        with col_s:
            start_date = st.date_input("Start", value=today - timedelta(days=365 * 2))
        with col_e:
            end_date = st.date_input("End", value=today)

    st.divider()
    st.subheader("🛠 Technical Overlays")
    show_sma20 = st.checkbox("20-Day SMA", value=True)
    show_sma50 = st.checkbox("50-Day SMA", value=True)
    show_ema20 = st.checkbox("20-Day EMA", value=False)
    show_bollinger = st.checkbox("Bollinger Bands (20, 2)", value=False)
    show_volume = st.checkbox("Trading Volume", value=True)

    st.divider()
    load_button = st.button("🔄 Refresh Data & Analysis", use_container_width=True, type="primary")

# ======================================================
# DATA FETCHING & PREPROCESSING
# ======================================================

if "current_ticker" not in st.session_state or st.session_state.current_ticker != ticker or load_button:
    st.session_state.current_ticker = ticker
    if "ai_results" in st.session_state:
        del st.session_state["ai_results"]

with st.spinner(f"📡 Ingesting market feed for {ticker}..."):
    df = load_stock_data(ticker, start_date, end_date)
    profile = get_stock_profile(ticker)

if df.empty or len(df) < 30:
    st.error(f"❌ Insufficient or no data found for ticker '{ticker}'. Please verify the symbol or expand the date range.")
    st.stop()

# Compute all technical indicators
df = add_indicators(df)

currency_symbol = profile.get("currency_symbol", "$")
curr_price = float(df['Close'].iloc[-1])
prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else curr_price
day_change = curr_price - prev_close
day_change_pct = (day_change / prev_close) * 100.0

# Initial Stock Position Analysis (without future forecast until AI tab is run)
future_preds_cached = st.session_state.get("ai_results", {}).get("future_predictions", None)
position_data = analyze_stock_position(df, future_predictions=future_preds_cached)

# Check if asset has valid volume data (Mutual Funds often have zero volume)
has_volume = 'Volume' in df.columns and df['Volume'].sum() > 0

# ======================================================
# TOP KPI BANNER
# ======================================================

st.markdown(f"## 📈 {selected_name} <span style='color:#64748b; font-size:1.2rem;'>({ticker})</span>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Current Price / NAV",
    f"{currency_symbol}{curr_price:,.2f}",
    delta=f"{day_change:+,.2f} ({day_change_pct:+.2f}%)"
)

y_high = profile.get("year_high") or float(df['High'].tail(252).max())
y_low = profile.get("year_low") or float(df['Low'].tail(252).min())
kpi2.metric(
    "52-Week High",
    f"{currency_symbol}{y_high:,.2f}"
)

kpi3.metric(
    "52-Week Low",
    f"{currency_symbol}{y_low:,.2f}"
)

mcap = profile.get("market_cap")
if mcap and mcap > 1e12:
    mcap_str = f"{currency_symbol}{mcap/1e12:.2f}T"
elif mcap and mcap > 1e9:
    mcap_str = f"{currency_symbol}{mcap/1e9:.2f}B"
elif mcap and mcap > 1e7 and currency_symbol == "₹":
    mcap_str = f"₹{mcap/1e7:.2f} Cr"
else:
    mcap_str = "N/A (Fund NAV)"

kpi4.metric(
    "Market Cap / AUM",
    mcap_str
)

with kpi5:
    st.markdown("<p style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 4px;'>Position Signal</p>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='background-color:{position_data['color']}22; border:1px solid {position_data['color']}; "
        f"color:{position_data['color']}; font-weight:700; padding:10px 14px; border-radius:10px; text-align:center; font-size:1.1rem;'>"
        f"{position_data['badge']}</div>",
        unsafe_allow_html=True
    )

st.write("")

# ======================================================
# INTERACTIVE WORKSPACE TABS
# ======================================================

tab_charts, tab_position, tab_ai, tab_data = st.tabs([
    "📊 Technical Analysis Suite",
    "🎯 Stock Position & Risk Management",
    "🤖 AI LSTM Forecast & Evaluation",
    "📄 Historical Data & Export"
])

# ------------------------------------------------------
# TAB 1: TECHNICAL ANALYSIS SUITE
# ------------------------------------------------------
with tab_charts:
    # Determine rows based on volume selection and actual volume presence
    include_volume = show_volume and has_volume

    if include_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25]
        )
    else:
        fig = go.Figure()

    # Candlestick Trace
    candlestick_trace = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price / NAV'
    )

    if include_volume:
        fig.add_trace(candlestick_trace, row=1, col=1)
    else:
        fig.add_trace(candlestick_trace)

    # Technical Indicator Overlays
    if show_sma20 and 'SMA_20' in df.columns:
        sma20_trace = go.Scatter(
            x=df.index, y=df['SMA_20'],
            mode='lines', name='20 SMA',
            line=dict(color='#f59e0b', width=1.5)
        )
        fig.add_trace(sma20_trace, row=1, col=1 if include_volume else None)

    if show_sma50 and 'SMA_50' in df.columns:
        sma50_trace = go.Scatter(
            x=df.index, y=df['SMA_50'],
            mode='lines', name='50 SMA',
            line=dict(color='#3b82f6', width=1.5)
        )
        fig.add_trace(sma50_trace, row=1, col=1 if include_volume else None)

    if show_ema20 and 'EMA_20' in df.columns:
        ema20_trace = go.Scatter(
            x=df.index, y=df['EMA_20'],
            mode='lines', name='20 EMA',
            line=dict(color='#06b6d4', width=1.5, dash='dot')
        )
        fig.add_trace(ema20_trace, row=1, col=1 if include_volume else None)

    if show_bollinger and 'BB_high' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_high'],
            mode='lines', name='BB Upper',
            line=dict(color='rgba(148, 163, 184, 0.4)', width=1)
        ), row=1, col=1 if include_volume else None)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_low'],
            mode='lines', name='BB Lower',
            line=dict(color='rgba(148, 163, 184, 0.4)', width=1),
            fill='tonexty', fillcolor='rgba(148, 163, 184, 0.05)'
        ), row=1, col=1 if include_volume else None)

    # Volume Subplot
    if include_volume:
        colors = ['#10b981' if c >= o else '#ef4444' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.8
        ), row=2, col=1)

    # Layout: Ample top margin & separate legend placement so modebar never overlaps title/legend
    fig.update_layout(
        title=dict(
            text=f"<b>{selected_name} ({ticker})</b> - Historical Price & Technical Indicator Overlays",
            font=dict(size=16, color="#f8fafc"),
            x=0.01,
            xanchor="left",
            y=0.98
        ),
        template="plotly_dark",
        height=620,
        xaxis_rangeslider_visible=False,
        yaxis_title=f"Price ({currency_symbol})",
        hovermode="x unified",
        margin=dict(t=80, b=40, l=50, r=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1
        )
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Oscillators: RSI and MACD in side-by-side columns
    col_rsi, col_macd = st.columns(2)

    with col_rsi:
        if 'RSI' in df.columns:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI'],
                mode='lines', name='RSI (14)',
                line=dict(color='#a855f7', width=2)
            ))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
            rsi_fig.update_layout(
                title=dict(
                    text=f"<b>{selected_name} ({ticker})</b> - Relative Strength Index (RSI 14): {df['RSI'].iloc[-1]:.1f}",
                    font=dict(size=13, color="#f8fafc"),
                    x=0.02,
                    xanchor="left"
                ),
                template="plotly_dark",
                height=280,
                yaxis=dict(range=[0, 100], title="RSI"),
                margin=dict(t=50, b=30, l=40, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02)
            )
            st.plotly_chart(rsi_fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col_macd:
        if 'MACD' in df.columns:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                mode='lines', name='MACD',
                line=dict(color='#38bdf8', width=1.5)
            ))
            macd_fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD_signal'],
                mode='lines', name='Signal',
                line=dict(color='#f97316', width=1.5)
            ))
            hist_colors = ['#10b981' if v >= 0 else '#ef4444' for v in df['MACD_diff']]
            macd_fig.add_trace(go.Bar(
                x=df.index, y=df['MACD_diff'],
                name='Histogram',
                marker_color=hist_colors
            ))
            macd_fig.update_layout(
                title=dict(
                    text=f"<b>{selected_name} ({ticker})</b> - Moving Average Convergence Divergence (MACD)",
                    font=dict(size=13, color="#f8fafc"),
                    x=0.02,
                    xanchor="left"
                ),
                template="plotly_dark",
                height=280,
                margin=dict(t=50, b=30, l=40, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02)
            )
            st.plotly_chart(macd_fig, use_container_width=True, config=PLOTLY_CONFIG)

# ------------------------------------------------------
# TAB 2: STOCK POSITION & RISK ENGINE
# ------------------------------------------------------
with tab_position:
    st.subheader("🎯 Institutional Stock Position & Risk Management")
    st.markdown("Algorithmic signal synthesis combining technical indicators, momentum, volatility, and AI projections.")

    # Main Position Card
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left: 6px solid {position_data['color']};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h3 style="margin:0; color:{position_data['color']};">{position_data['badge']}</h3>
                    <p style="margin:6px 0 0 0; color:#e2e8f0; font-size:1.05rem;">{position_data['action']}</p>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.9rem; color:#94a3b8;">Consensus Score:</span>
                    <h2 style="margin:0; color:{position_data['color']};">{position_data['score']:+.2f} / 4.5</h2>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Levels Cards
    pos_c1, pos_c2, pos_c3, pos_c4 = st.columns(4)

    pos_c1.metric(
        "Suggested Entry Zone",
        f"{currency_symbol}{position_data['entry_zone'][0]:,.2f} - {currency_symbol}{position_data['entry_zone'][1]:,.2f}",
        help="Optimal execution range near current market price."
    )

    pos_c2.metric(
        "Stop Loss (ATR 1.5x)",
        f"{currency_symbol}{position_data['stop_loss']:,.2f}",
        delta=f"-{currency_symbol}{position_data['risk_per_share']:.2f}",
        delta_color="inverse",
        help="Volatility-adjusted protective stop-loss based on Average True Range."
    )

    pos_c3.metric(
        "Target Price (Take Profit)",
        f"{currency_symbol}{position_data['target_price']:,.2f}",
        delta=f"+{currency_symbol}{position_data['reward_per_share']:.2f}",
        help="Calculated take-profit target aligned with upside resistance or AI forecast."
    )

    pos_c4.metric(
        "Risk / Reward Ratio",
        f"1 : {position_data['rr_ratio']:.2f}",
        help="Expected payout per unit of capital risked (aim for >= 1:2.0)."
    )

    st.divider()

    # 52-Week Range Bar
    st.markdown("#### ⏳ 52-Week Price Cycle Position")
    st.progress(min(1.0, max(0.0, position_data['pos_52w'] / 100.0)))
    col_low, col_pos, col_high = st.columns([1, 2, 1])
    col_low.markdown(f"<small style='color:#94a3b8;'>52W Low: <b>{currency_symbol}{position_data['low_52']:,.2f}</b></small>", unsafe_allow_html=True)
    col_pos.markdown(f"<div style='text-align:center;'><small style='color:#38bdf8;'>Current Price sits at <b>{position_data['pos_52w']:.1f}%</b> of 52-Week Range</small></div>", unsafe_allow_html=True)
    col_high.markdown(f"<div style='text-align:right;'><small style='color:#94a3b8;'>52W High: <b>{currency_symbol}{position_data['high_52']:,.2f}</b></small></div>", unsafe_allow_html=True)

    st.divider()

    # Breakdown Table & Interactive Position Sizing Calculator
    col_factors, col_sizing = st.columns([1.2, 1])

    with col_factors:
        st.markdown("#### 📋 Signal Factor Breakdown")
        breakdown_df = pd.DataFrame([
            {"Factor": k, "State & Assessment": v}
            for k, v in position_data["breakdown"].items()
        ])
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    with col_sizing:
        st.markdown("#### 💰 Interactive Position Sizing Calculator")
        with st.container(border=True):
            capital_input = st.number_input(
                f"Trading Account Capital ({currency_symbol})",
                min_value=100.0,
                max_value=100000000.0,
                value=10000.0 if currency_symbol == "$" else 100000.0,
                step=1000.0
            )

            risk_pct_input = st.slider(
                "Max Risk Per Trade (% of Capital)",
                min_value=0.5,
                max_value=10.0,
                value=2.0,
                step=0.5,
                help="Recommended risk per trade is 1% to 2% to protect against drawdowns."
            )

            sizing = calculate_position_sizing(
                capital_input,
                risk_pct_input,
                position_data["current_price"],
                position_data["stop_loss"]
            )

            sc1, sc2 = st.columns(2)
            sc1.metric("Max Capital at Risk", f"{currency_symbol}{sizing['max_risk_amount']:,.2f}")
            sc2.metric("Recommended Units / Shares", f"{sizing['recommended_shares']} units")

            sc3, sc4 = st.columns(2)
            sc3.metric("Total Position Allocation", f"{currency_symbol}{sizing['total_position_value']:,.2f}")
            sc4.metric("Portfolio Exposure", f"{sizing['portfolio_allocation_pct']:.1f}%")

# ------------------------------------------------------
# TAB 3: AI LSTM FORECAST & EVALUATION
# ------------------------------------------------------
with tab_ai:
    st.subheader("🤖 LSTM Deep Learning Forecasting & Out-of-Sample Validation")
    st.markdown("""
    This deep learning model utilizes stacked Long Short-Term Memory (LSTM) layers with Dropout regularization.
    Evaluation is conducted strictly **chronologically** (the final 20% of historical trading days form the out-of-sample forward test set, strictly eliminating data leakage).
    """)

    ai_btn = st.button("🚀 Train Model & Generate 30-Day Forecast", type="primary")

    if ai_btn or "ai_results" in st.session_state:
        if ai_btn or "ai_results" not in st.session_state:
            try:
                with st.spinner(f"🧠 Training Deep Learning LSTM Network for {ticker} (Chronological Split)..."):
                    dates_test, actual, predicted, future_predictions, metrics = predict_stock(df)
                    st.session_state["ai_results"] = {
                        "dates_test": dates_test,
                        "actual": actual,
                        "predicted": predicted,
                        "future_predictions": future_predictions,
                        "metrics": metrics
                    }
                    st.rerun()
            except Exception as e:
                st.error(f"❌ AI Prediction failed: {str(e)}")
                st.stop()

        # Retrieve cached results
        ai_res = st.session_state["ai_results"]
        dates_test = ai_res["dates_test"]
        actual = ai_res["actual"]
        predicted = ai_res["predicted"]
        future_predictions = ai_res["future_predictions"]
        metrics = ai_res["metrics"]

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Directional Accuracy",
            f"{metrics['directional_accuracy']:.1f}%",
            help="Percentage of out-of-sample days the model correctly forecasted whether the next close would be UP or DOWN."
        )

        m2.metric(
            "Model Accuracy (100 - MAPE)",
            f"{metrics['accuracy']:.2f}%",
            help="Mean Absolute Percentage Error over out-of-sample forward test period."
        )

        m3.metric(
            "RMSE",
            f"{currency_symbol}{metrics['rmse']:.2f}",
            help="Root Mean Squared Error in currency units."
        )

        future_target = float(future_predictions[-1])
        last_price = float(actual[-1])
        diff_price = future_target - last_price
        pct_proj = (diff_price / last_price) * 100.0

        m4.metric(
            "30-Day Forecast Target",
            f"{currency_symbol}{future_target:,.2f}",
            delta=f"{diff_price:+,.2f} ({pct_proj:+.1f}%)"
        )

        st.divider()

        # Out-of-Sample Test Evaluation Chart
        prediction_chart = go.Figure()
        prediction_chart.add_trace(go.Scatter(
            x=dates_test, y=actual,
            mode='lines', name='Actual Price / NAV',
            line=dict(color='#3b82f6', width=2)
        ))
        prediction_chart.add_trace(go.Scatter(
            x=dates_test, y=predicted,
            mode='lines', name='LSTM Predicted (Forward Test)',
            line=dict(color='#f97316', width=2, dash='dot')
        ))
        prediction_chart.update_layout(
            title=dict(
                text=f"<b>{selected_name} ({ticker})</b> - Out-of-Sample Historical Test Performance (LSTM Model)",
                font=dict(size=15, color="#f8fafc"),
                x=0.01,
                xanchor="left",
                y=0.98
            ),
            template="plotly_dark",
            height=460,
            xaxis_title="Date",
            yaxis_title=f"Price ({currency_symbol})",
            hovermode="x unified",
            margin=dict(t=70, b=30, l=50, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="#334155",
                borderwidth=1
            )
        )
        st.plotly_chart(prediction_chart, use_container_width=True, config=PLOTLY_CONFIG)

        st.divider()

        # 30-Day Forward Autoregressive Forecast Chart
        last_date = pd.to_datetime(df.index[-1])
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(future_predictions))

        future_chart = go.Figure()
        future_plot_dates = [last_date] + list(future_dates)
        future_plot_prices = [last_price] + list(future_predictions)

        future_chart.add_trace(go.Scatter(
            x=future_plot_dates,
            y=future_plot_prices,
            mode='lines+markers',
            name='30-Day Forecast Trajectory',
            line=dict(color='#10b981' if diff_price >= 0 else '#ef4444', width=3),
            marker=dict(size=4)
        ))

        future_chart.update_layout(
            title=dict(
                text=f"<b>{selected_name} ({ticker})</b> - 30-Day Forward Autoregressive Price Projection",
                font=dict(size=15, color="#f8fafc"),
                x=0.01,
                xanchor="left",
                y=0.98
            ),
            template="plotly_dark",
            height=460,
            xaxis_title="Future Trading Date",
            yaxis_title=f"Forecast Price ({currency_symbol})",
            hovermode="x unified",
            margin=dict(t=70, b=30, l=50, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="#334155",
                borderwidth=1
            )
        )

        if diff_price >= 0:
            money_showers()
        else:
            money_shower_loss()

        st.plotly_chart(future_chart, use_container_width=True, config=PLOTLY_CONFIG)

# ------------------------------------------------------
# TAB 4: HISTORICAL DATA & EXPORT
# ------------------------------------------------------
with tab_data:
    st.subheader(f"📄 {selected_name} ({ticker}) - Historical Data & Indicator Export")
    st.markdown("Review the raw historical prices, OHLCV records, and engineered indicators.")

    display_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 'MACD', 'ATR'] if c in df.columns]
    st.dataframe(df[display_cols].tail(50), use_container_width=True)

    csv_data = df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Historical & Indicator Data as CSV",
        data=csv_data,
        file_name=f"{ticker}_technical_data_{date.today().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary"
    )
