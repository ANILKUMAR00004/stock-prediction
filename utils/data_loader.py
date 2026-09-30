import yfinance as yf
import pandas as pd

def load_stock_data(ticker, start_date, end_date):
    """
    Downloads stock data from Yahoo Finance and normalizes column headers.
    """
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(col).strip() for col in df.columns]
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def get_stock_profile(ticker):
    """
    Pulls live fast_info & fundamental data for the ticker.
    """
    try:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        last_price = getattr(fast, 'last_price', None)
        prev_close = getattr(fast, 'previous_close', None)
        mcap = getattr(fast, 'market_cap', None)
        y_high = getattr(fast, 'year_high', None)
        y_low = getattr(fast, 'year_low', None)
        currency = getattr(fast, 'currency', 'USD')

        change = (last_price - prev_close) if (last_price and prev_close) else 0.0
        pct_change = (change / prev_close * 100.0) if (prev_close and prev_close > 0) else 0.0

        currency_symbol = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") or currency == "INR" else "$"

        return {
            "last_price": last_price,
            "prev_close": prev_close,
            "change": change,
            "pct_change": pct_change,
            "market_cap": mcap,
            "year_high": y_high,
            "year_low": y_low,
            "currency": currency,
            "currency_symbol": currency_symbol
        }
    except Exception:
        currency_symbol = "₹" if ticker.endswith(".NS") or ticker.endswith(".BO") else "$"
        return {
            "last_price": None,
            "prev_close": None,
            "change": 0.0,
            "pct_change": 0.0,
            "market_cap": None,
            "year_high": None,
            "year_low": None,
            "currency": "USD",
            "currency_symbol": currency_symbol
        }
