# ui/trading_signals_dashboard.py
"""
Trading Signals Dashboard (Streamlit)
- Fully local DecisionAgent integration
- Robust favorites loader (handles empty/corrupt JSON)
- Handles yfinance multi-index output
- Generates signals using signals.signal_engine
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json
import os
import sys
from datetime import datetime, timedelta
from signals.signal_engine import generate_signals

# Add repo root to sys.path for local agent import
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from agents.decision_agent import DecisionAgent

st.set_page_config(page_title="Trading Signals Dashboard", layout="wide")

CONFIG_DIR = "config"
CONFIG_FAV = os.path.join(CONFIG_DIR, "favorites.json")
DEFAULT_FAVS = ["AAPL", "MSFT", "TSLA"]

# Initialize a single DecisionAgent instance
agent_instance = DecisionAgent()


def save_favorites(favs):
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)
    tmp_file = CONFIG_FAV + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump({"favorites": favs}, f, indent=2)
    os.replace(tmp_file, CONFIG_FAV)


def load_favorites():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

    if not os.path.exists(CONFIG_FAV):
        save_favorites(DEFAULT_FAVS)
        return DEFAULT_FAVS

    try:
        with open(CONFIG_FAV, "r", encoding="utf-8") as f:
            data = json.load(f)
        favs = data.get("favorites", DEFAULT_FAVS)
        if not isinstance(favs, list):
            raise ValueError("favorites is not a list")
        favs = [s.strip().upper() for s in favs if isinstance(s, str) and s.strip()]
        return favs if favs else DEFAULT_FAVS
    except Exception:
        save_favorites(DEFAULT_FAVS)
        return DEFAULT_FAVS


def fetch_price_data(symbol, days=180):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)

        if df is None or df.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        # Fix multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for col_tuple in df.columns:
                if isinstance(col_tuple, tuple):
                    if col_tuple[1] in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                        new_cols.append(col_tuple[1])
                    else:
                        new_cols.append(col_tuple[0])
                else:
                    new_cols.append(col_tuple)
            df.columns = new_cols

        # Ensure "Close" column exists
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                df["Close"] = pd.NA

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["Close"])
        return df

    except Exception as e:
        print("fetch_price_data error:", e)
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])


def call_decision_agent_local(symbol, df, signals):
    """
    Call local DecisionAgent.decide() with market_state.
    Returns a dict (parsed JSON).
    """
    try:
        recent_prices = df["Close"].tail(30).tolist()
        indicators_payload = {
            "rsi": signals.get("details", {}).get("rsi_14d"),
            "macd": signals.get("details", {}).get("ma_diff"),
            "signal": signals.get("signal", "HOLD"),
        }

        market_state = {
            "ticker": symbol,
            "recent_prices": recent_prices,
            "indicators": indicators_payload
        }

        raw_response = agent_instance.decide(market_state)

        # Parse JSON safely
        try:
            result = json.loads(raw_response)
        except Exception:
            if isinstance(raw_response, dict):
                result = raw_response
            else:
                result = {"raw_response": raw_response}

        return result

    except Exception as e:
        return {"error": str(e), "raw": repr(e)}


def main():
    st.title("Trading Signals Dashboard")
    st.markdown("Track favorite stocks, view generated signals, and analyze with local Decision Agent.")

    favorites = load_favorites()

    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.header("Favorites")
        new_sym = st.text_input("Add ticker (e.g. AAPL)", key="add_ticker_input")
        if st.button("Add"):
            sym = new_sym.strip().upper()
            if sym and sym not in favorites:
                favorites.append(sym)
                save_favorites(favorites)
                st.experimental_rerun()

        rem = st.selectbox("Remove ticker", [""] + favorites, key="remove_ticker")
        if st.button("Remove"):
            if rem in favorites:
                favorites = [s for s in favorites if s != rem]
                save_favorites(favorites)
                st.experimental_rerun()

        st.markdown("**Current favorites**")
        st.write(favorites)

        st.markdown("---")
        st.markdown("**Settings**")
        days = st.slider("Days of history to fetch", min_value=30, max_value=1095, value=180, step=30)

    with right_col:
        st.header("Market View")
        if not favorites:
            st.warning("No favorites available. Add a ticker on the left.")
            return

        selected = st.selectbox("Select favorite ticker", favorites, index=0)
        if not selected:
            st.info("Select a ticker to view its data.")
            return

        try:
            with st.spinner(f"Fetching price data for {selected} ..."):
                df = fetch_price_data(selected, days=days)

            if df.empty:
                st.error(f"No price data found for {selected}")
                return

            latest_close_val = df["Close"].iloc[-1]
            if isinstance(latest_close_val, (pd.Series, np.ndarray)):
                latest_close_val = np.asarray(latest_close_val).ravel()[0]
            latest_close = float(latest_close_val)

            st.subheader(f"{selected} — latest close: {latest_close:.2f}")
            st.line_chart(df["Close"])

            # Generate signals
            sig = generate_signals(df)

            st.metric("Signal", sig.get("signal", "HOLD"))
            conf = sig.get("confidence", 0.0)
            try:
                st.metric("Confidence", f"{float(conf) * 100:.1f}%")
            except Exception:
                st.metric("Confidence", f"{conf}")

            st.markdown("**Signal details**")
            st.json(sig.get("details", {}))

            # Local Decision Agent button
            if st.button("Ask Decision Agent for explanation"):
                with st.spinner("Analyzing with local decision agent..."):
                    agent_resp = call_decision_agent_local(selected, df, sig)

                if "error" in agent_resp:
                    st.error(agent_resp)
                else:
                    st.success(agent_resp.get("decision", agent_resp.get("action", "No decision")))
                    st.json(agent_resp)

        except Exception as e:
            st.error("An unexpected error occurred while rendering the view.")
            st.text(str(e))


if __name__ == "__main__":
    main()
