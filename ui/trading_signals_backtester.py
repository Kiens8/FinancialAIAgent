import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os

# Try to import vectorbt for the backtesting engine
try:
    import vectorbt as vbt
except ImportError:
    st.error("Library 'vectorbt' is missing. Please install it using: pip install vectorbt")
    st.stop()

st.set_page_config(layout='wide', page_title='Financial AI — Strategy Backtest')
st.title('Financial AI — Signal Engine Strategy Backtester')

# ------------------------------
# Config & Favorites Loader
# ------------------------------
CONFIG_DIR = "config"
FAV_FILE = os.path.join(CONFIG_DIR, "favorites.json")
DEFAULT_FAVS = ["AAPL", "MSFT", "TSLA"]

def load_favorites():
    if not os.path.exists(FAV_FILE):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(FAV_FILE, "w") as f:
            json.dump({"favorites": DEFAULT_FAVS}, f, indent=2)
        return DEFAULT_FAVS

    try:
        with open(FAV_FILE, "r") as f:
            data = json.load(f)
        favs = data.get("favorites", DEFAULT_FAVS)
        favs = [s.strip().upper() for s in favs if s.strip()]
        return favs if favs else DEFAULT_FAVS
    except Exception as e:
        print(f"Error reading favorites.json: {e}")
        return DEFAULT_FAVS

# ------------------------------
# Data Fetching (Synced with signal_engine)
# ------------------------------
def safe_download(tickers, period="1y", interval="1d"):
    """
    Downloads data for one or multiple tickers.
    Force auto_adjust=False to match signal_engine logic.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    data = {}
    
    for t in tickers:
        try:
            df = yf.download(
                t, 
                period=period, 
                interval=interval, 
                progress=False, 
                auto_adjust=False # CRITICAL: Match main.py/dashboard
            )
            
            if df is None or df.empty:
                continue

            # Clean MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                new_cols = []
                for col_tuple in df.columns:
                    if isinstance(col_tuple, tuple):
                        if col_tuple[1] in ["Open", "High", "Low", "Close", "Volume"]:
                            new_cols.append(col_tuple[1])
                        else:
                            new_cols.append(col_tuple[0])
                    else:
                        new_cols.append(col_tuple)
                df.columns = new_cols
            
            # Standardize
            df.columns = [c.capitalize() for c in df.columns]
            
            # Drop NA on Close to match Dashboard looseness
            if "Close" in df.columns:
                df = df.dropna(subset=["Close"])
                data[t] = df["Close"] # We primarily need Close for this strategy
                
        except Exception as e:
            print(f"Error downloading {t}: {e}")

    if not data:
        return pd.DataFrame()
    
    # Create Batch DataFrame
    df_combined = pd.DataFrame(data)
    
    # ---------------------------------------------------------
    # 🔥 FIX FOR BATCH MODE (Stocks + Crypto)
    # ---------------------------------------------------------
    # When mixing Crypto (7 days) and Stocks (5 days), Stocks get NaNs on weekends.
    # This breaks Rolling MAs and RSI calculations.
    # We forward fill (ffill) so Saturday/Sunday prices = Friday's close.
    df_combined.ffill(inplace=True)
    
    # Drop initial NaNs (e.g., if one stock listed later than others)
    df_combined.dropna(how='all', inplace=True)
    
    return df_combined

# ------------------------------
# Indicator Logic (Replicated from signal_engine.py)
# ------------------------------
def calculate_indicators(close_series):
    """
    Vectorized calculation matching signal_engine.py exactly.
    Uses Rolling Mean (SMA) for MA and RSI (not EMA).
    """
    # Constants from signal_engine.py
    SHORT_W = 20
    LONG_W = 50
    MOM_W = 5
    RSI_W = 14
    
    # 1. Moving Averages (Simple)
    short_ma = close_series.rolling(window=SHORT_W, min_periods=1).mean()
    long_ma = close_series.rolling(window=LONG_W, min_periods=1).mean()
    
    # 2. Momentum (Pct Change)
    momentum = close_series.pct_change(periods=MOM_W)
    
    # 3. RSI (Simple Moving Average method, matching signal_engine)
    delta = close_series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # signal_engine uses rolling mean for RSI, not Wilder's
    ma_up = up.rolling(window=RSI_W, min_periods=RSI_W).mean()
    ma_down = down.rolling(window=RSI_W, min_periods=RSI_W).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    
    return short_ma, long_ma, momentum, rsi

def run_strategy(close_series, fees=0.001):
    """
    Applies the Buy/Sell rules from signal_engine.py
    """
    short_ma, long_ma, momentum, rsi = calculate_indicators(close_series)
    
    # --- BUY RULE ---
    # signal_engine: if (short > long) and (mom > 0): if rsi < 70: BUY
    entries = (short_ma > long_ma) & (momentum > 0) & (rsi < 70)
    
    # --- SELL RULE ---
    # signal_engine: if (short < long) and (mom < 0): if rsi > 30: SELL
    exits = (short_ma < long_ma) & (momentum < 0) & (rsi > 30)
    
    # Run Backtest using VectorBT
    # init_cash=10000 is arbitrary for percentage calculations
    pf = vbt.Portfolio.from_signals(
        close_series, 
        entries, 
        exits, 
        fees=fees, 
        freq='D',
        init_cash=10000
    )
    return pf

# ------------------------------
# Helper: Clean Stats Display
# ------------------------------
def display_clean_stats(pf):
    """
    Fetches stats, replaces NaNs with '-' or 0, and displays as a nice table.
    """
    stats = pf.stats()
    df_stats = stats.to_frame(name="Value")
    df_stats = df_stats.astype(object) # Convert to object to hold strings
    df_stats.fillna("-", inplace=True) # Replace NaNs
    st.table(df_stats)

# ------------------------------
# UI Sidebar
# ------------------------------
with st.sidebar:
    st.header('Backtest Settings')
    favorites = load_favorites()
    
    mode = st.radio("Backtest Mode", ["Single Ticker", "All Favorites (Batch)"])
    
    if mode == "Single Ticker":
        selection_mode = st.radio("Ticker Source", ["Favorites", "Custom"], horizontal=True)
        if selection_mode == "Favorites":
            ticker = st.selectbox('Select Ticker', favorites)
        else:
            ticker = st.text_input('Type Ticker', value='AAPL').strip().upper()
    else:
        st.info(f"Will test: {', '.join(favorites)}")
        ticker = favorites # List for batch

    period = st.selectbox('Period', ['6mo','1y','2y','5y', 'max'], index=2)
    fees = st.number_input('Trading Fees (decimal)', value=0.0005, step=0.0001, format="%.4f")
    
    st.markdown("---")
    st.markdown("### Strategy Logic")
    st.caption("Rules synced with `signal_engine.py`")
    st.markdown("""
    - **Short MA**: 20 SMA
    - **Long MA**: 50 SMA
    - **Momentum**: 5-day %
    - **RSI**: 14-day (SMA based)
    
    **Buy**: Short > Long AND Mom > 0 AND RSI < 70  
    **Sell**: Short < Long AND Mom < 0 AND RSI > 30
    """)

# ------------------------------
# Main Execution
# ------------------------------
if st.button('Run Backtest'):
    
    # 1. Fetch Data
    with st.spinner('Fetching market data...'):
        if mode == "Single Ticker":
            prices_df = safe_download([ticker], period)
        else:
            prices_df = safe_download(favorites, period)
            
    if prices_df.empty:
        st.error("No data found. Please check your tickers or internet connection.")
        st.stop()

    # 2. Run Strategy
    try:
        # Determine if we have one column (Series) or multiple (DataFrame)
        if mode == "Single Ticker":
            if ticker in prices_df.columns:
                close_data = prices_df[ticker]
            else:
                close_data = prices_df.iloc[:, 0]
        else:
            close_data = prices_df # DataFrame of multiple columns
            
        pf = run_strategy(close_data, fees=fees)
        
        # 3. Display Results
        if mode == "Single Ticker":
            st.subheader(f"Performance: {ticker}")
            
            total_return = pf.total_return()
            max_drawdown = pf.max_drawdown()
            win_rate = pf.trades.win_rate()
            trade_count = pf.trades.count()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Return", f"{total_return:.2%}")
            col3.metric("Max Drawdown", f"{max_drawdown:.2%}")

            if trade_count == 0:
                col2.metric("Win Rate", "N/A (0 Trades)")
            else:
                col2.metric("Win Rate", f"{win_rate:.2%}")
            
            st.subheader("Equity Curve")
            st.plotly_chart(pf.plot(title=f"Equity - {ticker}"), use_container_width=True)
            
            with st.expander("View Trade Log", expanded=True):
                if trade_count > 0:
                    st.dataframe(pf.orders.records_readable)
                else:
                    st.info("No trades were executed.")

            with st.expander("View Statistics"):
                display_clean_stats(pf)

        else:
            # Batch Results
            st.subheader("Batch Performance Comparison")
            
            # Gather metrics
            # Replace Infinity with 0.0 or NaN to prevent Arrow Serialization Error
            total_ret = pf.total_return().replace([np.inf, -np.inf], 0.0)
            sharpe = pf.sharpe_ratio().replace([np.inf, -np.inf], 0.0)
            max_dd = pf.max_drawdown().replace([np.inf, -np.inf], 0.0)
            
            metrics_df = pd.DataFrame({
                "Total Return": total_ret,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_dd
            })
            
            # Sort by Total Return
            metrics_df = metrics_df.sort_values("Total Return", ascending=False)
            
            # Formatting
            st.dataframe(
                metrics_df.style
                .format("{:.2%}", subset=["Total Return", "Max Drawdown"])
                .format("{:.2f}", subset=["Sharpe Ratio"])
                .background_gradient(cmap="RdYlGn", subset=["Total Return"])
            )
            
            st.subheader("Cumulative Returns Comparison")
            st.line_chart(pf.value())
            
            best_ticker = metrics_df.index[0]
            best_ret = metrics_df.iloc[0]["Total Return"]
            st.success(f"🏆 Best Performer: **{best_ticker}** with {best_ret:.2%} return")

    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")