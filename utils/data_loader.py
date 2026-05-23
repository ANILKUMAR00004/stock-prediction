import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    # FIX MULTI INDEX COLUMNS
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # RESET COLUMN NAMES
    df.columns = [str(col).strip() for col in df.columns]

    return df