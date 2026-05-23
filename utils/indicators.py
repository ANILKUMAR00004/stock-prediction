import ta
import pandas as pd

def add_indicators(df):

    # FORCE 1D SERIES
    close = pd.Series(df['Close']).squeeze()

    df['SMA_20'] = ta.trend.sma_indicator(
        close=close,
        window=20
    )

    df['EMA_20'] = ta.trend.ema_indicator(
        close=close,
        window=20
    )

    df['RSI'] = ta.momentum.rsi(
        close=close,
        window=14
    )

    return df