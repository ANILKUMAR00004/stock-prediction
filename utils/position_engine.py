import numpy as np
import pandas as pd

def analyze_stock_position(df, future_predictions=None):
    """
    Algorithmic consensus engine that evaluates AI forecast, RSI momentum,
    moving average trend, and MACD to generate trade signals and risk parameters.
    """
    current_price = float(df['Close'].iloc[-1])
    
    # 52-Week Range
    trailing_year = df.tail(252)
    high_52 = float(trailing_year['High'].max())
    low_52 = float(trailing_year['Low'].min())
    if high_52 > low_52:
        pos_52w = ((current_price - low_52) / (high_52 - low_52)) * 100.0
    else:
        pos_52w = 50.0

    # ATR for volatility-based risk management
    if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]):
        atr = float(df['ATR'].iloc[-1])
    else:
        atr = current_price * 0.025

    # 1. Moving Average Alignment Score
    ma_score = 0.0
    ma_reasons = []
    sma_20 = float(df['SMA_20'].iloc[-1]) if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]) else None
    sma_50 = float(df['SMA_50'].iloc[-1]) if 'SMA_50' in df.columns and not pd.isna(df['SMA_50'].iloc[-1]) else None
    ema_20 = float(df['EMA_20'].iloc[-1]) if 'EMA_20' in df.columns and not pd.isna(df['EMA_20'].iloc[-1]) else None

    if sma_20 and sma_50:
        if current_price > sma_20 > sma_50:
            ma_score += 1.0
            ma_reasons.append("Bullish Moving Average stack (Price > SMA20 > SMA50)")
        elif current_price < sma_20 < sma_50:
            ma_score -= 1.0
            ma_reasons.append("Bearish Moving Average stack (Price < SMA20 < SMA50)")
        elif current_price > sma_20:
            ma_score += 0.5
            ma_reasons.append("Trading above 20-Day SMA")
        else:
            ma_score -= 0.5
            ma_reasons.append("Trading below 20-Day SMA")
    elif sma_20:
        if current_price > sma_20:
            ma_score += 0.5
            ma_reasons.append("Trading above 20-Day SMA")
        else:
            ma_score -= 0.5
            ma_reasons.append("Trading below 20-Day SMA")

    # 2. RSI Momentum Score
    rsi_score = 0.0
    rsi_val = float(df['RSI'].iloc[-1]) if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]) else 50.0
    if rsi_val < 30:
        rsi_score += 1.0
        rsi_status = f"Oversold (RSI: {rsi_val:.1f}) - Potential mean reversion rebound"
    elif 30 <= rsi_val <= 45:
        rsi_score += 0.25
        rsi_status = f"Mild Oversold / Accumulation (RSI: {rsi_val:.1f})"
    elif 45 < rsi_val < 65:
        rsi_score += 0.75
        rsi_status = f"Healthy Bullish Momentum (RSI: {rsi_val:.1f})"
    elif 65 <= rsi_val <= 75:
        rsi_score -= 0.25
        rsi_status = f"Approaching Overbought (RSI: {rsi_val:.1f})"
    else:
        rsi_score -= 1.0
        rsi_status = f"Overbought (RSI: {rsi_val:.1f}) - Elevated pullback risk"

    # 3. MACD Momentum Score
    macd_score = 0.0
    macd_diff = float(df['MACD_diff'].iloc[-1]) if 'MACD_diff' in df.columns and not pd.isna(df['MACD_diff'].iloc[-1]) else 0.0
    macd_val = float(df['MACD'].iloc[-1]) if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]) else 0.0
    macd_sig = float(df['MACD_signal'].iloc[-1]) if 'MACD_signal' in df.columns and not pd.isna(df['MACD_signal'].iloc[-1]) else 0.0

    if macd_diff > 0 and macd_val > macd_sig:
        macd_score += 1.0
        macd_status = "Bullish MACD crossover (Histogram positive)"
    elif macd_diff < 0 and macd_val < macd_sig:
        macd_score -= 1.0
        macd_status = "Bearish MACD crossover (Histogram negative)"
    else:
        macd_status = "Neutral MACD momentum"

    # 4. AI LSTM Forecast Score
    ai_score = 0.0
    if future_predictions is not None and len(future_predictions) > 0:
        forecast_price = float(future_predictions[-1])
        pct_projected = ((forecast_price - current_price) / current_price) * 100.0
        if pct_projected >= 5.0:
            ai_score += 1.5
            ai_status = f"Strong Bullish Forecast (+{pct_projected:.1f}% in 30 days)"
        elif pct_projected >= 1.5:
            ai_score += 0.75
            ai_status = f"Moderate Bullish Forecast (+{pct_projected:.1f}% in 30 days)"
        elif pct_projected <= -5.0:
            ai_score -= 1.5
            ai_status = f"Strong Bearish Forecast ({pct_projected:.1f}% in 30 days)"
        elif pct_projected <= -1.5:
            ai_score -= 0.75
            ai_status = f"Moderate Bearish Forecast ({pct_projected:.1f}% in 30 days)"
        else:
            ai_status = f"Neutral Forecast ({pct_projected:+.1f}% in 30 days)"
    else:
        forecast_price = current_price * 1.04
        pct_projected = 4.0
        ai_status = "Model forecast pending (Run AI tab)"

    total_score = ma_score + rsi_score + macd_score + ai_score

    # Determine Consensus Recommendation
    if total_score >= 2.25:
        signal = "STRONG BUY"
        badge = "🟢 STRONG BUY"
        color = "#10b981"
        action = "High-conviction bullish setup. Look to accumulate long position."
    elif total_score >= 0.75:
        signal = "BUY"
        badge = "🟢 BUY"
        color = "#34d399"
        action = "Bullish momentum aligned. Long entry on minor pullbacks recommended."
    elif total_score <= -2.25:
        signal = "STRONG SELL"
        badge = "🔴 STRONG SELL"
        color = "#ef4444"
        action = "High downside momentum. Liquidate or hedge long exposure."
    elif total_score <= -0.75:
        signal = "SELL"
        badge = "🔴 SELL"
        color = "#f87171"
        action = "Bearish pressure detected. Consider trimming position or setting tight stops."
    else:
        signal = "HOLD"
        badge = "🟡 HOLD / NEUTRAL"
        color = "#f59e0b"
        action = "Consolidation zone. Maintain existing exposure and wait for breakout confirmation."

    # Trade Levels Calculation (Stop-Loss & Take-Profit Targets)
    entry_zone = (round(current_price * 0.995, 2), round(current_price * 1.005, 2))
    stop_loss = round(max(0.01, current_price - 1.5 * atr), 2)
    risk_per_share = max(0.01, current_price - stop_loss)

    if future_predictions is not None and len(future_predictions) > 0 and future_predictions[-1] > current_price:
        target_price = round(max(current_price + 2.0 * risk_per_share, float(future_predictions[-1])), 2)
    else:
        target_price = round(current_price + 2.0 * risk_per_share, 2)

    reward_per_share = max(0.01, target_price - current_price)
    rr_ratio = reward_per_share / risk_per_share

    return {
        "signal": signal,
        "badge": badge,
        "color": color,
        "action": action,
        "score": round(total_score, 2),
        "current_price": current_price,
        "high_52": high_52,
        "low_52": low_52,
        "pos_52w": round(pos_52w, 1),
        "atr": round(atr, 2),
        "entry_zone": entry_zone,
        "stop_loss": stop_loss,
        "target_price": target_price,
        "risk_per_share": round(risk_per_share, 2),
        "reward_per_share": round(reward_per_share, 2),
        "rr_ratio": round(rr_ratio, 2),
        "breakdown": {
            "Trend (Moving Averages)": ma_reasons[0] if ma_reasons else "Neutral Trend",
            "Momentum (RSI)": rsi_status,
            "Trend-Following (MACD)": macd_status,
            "AI 30-Day Forecast": ai_status
        }
    }

def calculate_position_sizing(account_capital, risk_pct, current_price, stop_loss):
    """
    Computes exact risk-controlled position sizing.
    """
    risk_capital = account_capital * (risk_pct / 100.0)
    risk_per_share = max(0.01, current_price - stop_loss)
    shares = int(risk_capital / risk_per_share)
    total_position_value = round(shares * current_price, 2)
    portfolio_pct = round((total_position_value / account_capital) * 100.0, 1)

    return {
        "max_risk_amount": round(risk_capital, 2),
        "recommended_shares": shares,
        "total_position_value": total_position_value,
        "portfolio_allocation_pct": portfolio_pct
    }
