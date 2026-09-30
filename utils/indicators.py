import ta
import pandas as pd

def add_indicators(df):
    """
    Computes key trend, momentum, volatility, and risk management indicators.
    """
    close = pd.Series(df['Close']).squeeze()
    high = pd.Series(df['High']).squeeze()
    low = pd.Series(df['Low']).squeeze()

    # Moving Averages (Trend)
    df['SMA_20'] = ta.trend.sma_indicator(close=close, window=20)
    df['SMA_50'] = ta.trend.sma_indicator(close=close, window=50)
    df['EMA_20'] = ta.trend.ema_indicator(close=close, window=20)

    # Momentum (RSI 14)
    df['RSI'] = ta.momentum.rsi(close=close, window=14)

    # MACD (12, 26, 9)
    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    # Bollinger Bands (20, 2)
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['BB_mid'] = bb.bollinger_mavg()

    # Average True Range (ATR 14) for volatility & stop loss
    df['ATR'] = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)

    return df
