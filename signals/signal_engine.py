# signals/signal_engine.py
"""
Signal engine with multiple explainable indicators:
- Moving average crossover (fast/slow)
- Momentum (pct change over N days)
- ATR (approx via true range)
- RSI (basic implementation)
- Volatility (rolling std)
Returns:
{
  "signal": "BUY"|"SELL"|"HOLD",
  "confidence": 0-1,
  "details": { ... indicators ... }
}
"""

import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period, min_periods=period).mean()
    ma_down = down.rolling(period, min_periods=period).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def generate_signals(df: pd.DataFrame) -> dict:
    try:
        if df is None or df.empty:
            return {"signal": "HOLD", "confidence": 0.0, "details": {"reason": "no data"}}

        close = df["Close"].dropna().astype(float)
        if close.empty:
            return {"signal": "HOLD", "confidence": 0.0, "details": {"reason": "no close prices"}}

        short_w = 20
        long_w = 50
        mom_w = 5
        rsi_w = 14

        short_ma = close.rolling(short_w, min_periods=1).mean()
        long_ma = close.rolling(long_w, min_periods=1).mean()
        momentum = close.pct_change(mom_w).iloc[-1] if len(close) > mom_w else 0.0
        vol = close.pct_change().rolling(10, min_periods=1).std().iloc[-1]
        latest_short = short_ma.iloc[-1]
        latest_long = long_ma.iloc[-1]
        latest_rsi = float(rsi(close, rsi_w).iloc[-1]) if len(close) >= rsi_w else float("nan")

        latest_atr = float(atr(df, 14).iloc[-1]) if "High" in df.columns and "Low" in df.columns else None

        signal = "HOLD"
        if (latest_short > latest_long) and (momentum > 0):
            if np.isnan(latest_rsi) or latest_rsi < 70:
                signal = "BUY"
            else:
                signal = "HOLD"
        elif (latest_short < latest_long) and (momentum < 0):
            if np.isnan(latest_rsi) or latest_rsi > 30:
                signal = "SELL"
            else:
                signal = "HOLD"

        ma_diff = abs(latest_short - latest_long) / (latest_long + 1e-9)

        conf = 0.0
        conf += min(1.0, abs(momentum) * 5)
        conf += min(1.0, ma_diff * 10)
        if vol < 0.02:
            conf += 0.1
        conf = float(min(1.0, conf))

        details = {
            "short_ma": float(latest_short),
            "long_ma": float(latest_long),
            "momentum_{}d".format(mom_w): float(momentum),
            "vol_10d": float(vol) if not np.isnan(vol) else None,
            "ma_diff": float(ma_diff),
            "rsi_{}d".format(rsi_w): float(latest_rsi) if not np.isnan(latest_rsi) else None,
            "atr_14d": float(latest_atr) if latest_atr is not None else None,
            "rule": "20/50 MA cross + 5d momentum + RSI filter"
        }

        return {"signal": signal, "confidence": conf, "details": details}

    except Exception as e:
        return {"signal": "HOLD", "confidence": 0.0, "details": {"error": str(e)}}
