import streamlit as st
import yfinance as yf
# Ensure the folder 'backtester' exists with '__init__.py' or is in python path
from backtester.vectorbt_backtester import (
    sma_crossover_backtest_single,
    sma_crossover_backtest_batch
)
import pandas as pd
import time
import numpy as np

st.set_page_config(layout='wide', page_title='Financial AI — Multi SMA Backtest')
st.title('Financial AI — SMA Multi-Run Backtester V2')

# ------------------------------
# Safe download function (FIXED for JSON Errors)
# ------------------------------

def safe_download(ticker, period="1y", interval="1d"):
    """
    Download with retries + fallbacks.
    Catches JSONDecodeError (Expecting value...) which happens when YF blocks requests.
    """
    # Normalize ticker
    ticker = ticker.strip().upper()
    
    for attempt in range(3):
        try:
            # Method 1: yf.Ticker.history (Preferred)
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                period=period, 
                interval=interval, 
                auto_adjust=True
            )
            
            # Clean MultiIndex (Common yfinance v0.2+ issue)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not df.empty:
                return _clean_columns(df)
                
        except (ValueError, OSError, Exception) as e:
            # ValueError covers the "Expecting value: line 1..." JSON error
            print(f"Attempt {attempt+1} (History) failed: {e}")
            pass
        time.sleep(1)

    # Method 2: yf.download (Fallback)
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False
        )
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=1, axis=1)
            except:
                df.columns = df.columns.get_level_values(0)

        if not df.empty:
            return _clean_columns(df)
    except Exception as e:
        print(f"Fallback download failed: {e}")

    # FINAL fallback: Synthetic Data
    st.warning(f"Yahoo Finance blocked access or failed for {ticker}. Using Synthetic Data for demo.")
    return _generate_synthetic_data()

def _clean_columns(df):
    """Standardize column names to Title Case (Close, Open, etc.)"""
    df.columns = [c.capitalize() for c in df.columns]
    return df

def _generate_synthetic_data():
    """Generates fake data so the app doesn't crash."""
    dates = pd.date_range(end=pd.Timestamp.today(), periods=200)
    # Create a random walk with drift
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, size=len(dates))
    price_path = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({"Close": price_path}, index=dates)
    df["Open"] = df["Close"]
    df["High"] = df["Close"] * 1.01
    df["Low"] = df["Close"] * 0.99
    df["Volume"] = 1000000
    return df

# ------------------------------
# UI Sidebar
# ------------------------------

with st.sidebar:
    st.header('Backtest Settings')
    ticker = st.text_input('Ticker', value='AAPL')
    period = st.selectbox('Period', ['6mo','1y','2y','5y', 'max'], index=1)
    fees = st.number_input('Fees & slippage (fraction)', value=0.0005, format="%.6f")

    mode = st.radio('Mode', ['Single run','Batch runs'])

    if mode == 'Single run':
        fast = st.number_input('Fast SMA', value=10, min_value=2, format='%d')
        slow = st.number_input('Slow SMA', value=50, min_value=3, format='%d')

    else:
        fast_list_raw = st.text_input('Fast Windows', value='5,10,15')
        slow_list_raw = st.text_input('Slow Windows', value='20,50')


# ------------------------------
# Run Button
# ------------------------------

if st.button('Run Backtest'):
    with st.spinner(f'Downloading data for {ticker}...'):
        df = safe_download(ticker, period)

    if df.empty:
        st.error(f"No data fetched for {ticker}. Please check the symbol.")
        st.stop()

    # Extract closing price
    # Handle case sensitivity (Close vs close)
    if 'Close' in df.columns:
        price = df['Close'].dropna()
    elif 'close' in df.columns:
        price = df['close'].dropna()
    else:
        st.error(f"Column 'Close' not found. Available columns: {df.columns.tolist()}")
        st.stop()

    # ------------------------------
    # Single Run Mode
    # ------------------------------
    if mode == 'Single run':
        fast = int(fast)
        slow = int(slow)
        
        if fast >= slow:
            st.error("Fast MA must be smaller than Slow MA")
            st.stop()

        pf = sma_crossover_backtest_single(price, fast, slow, fees=fees)

        # vectorbt normally exposes .equity instead of .value
        # Handling both generic object and vectorbt Portfolio object
        if hasattr(pf, "plot"):
             # If it's a real vectorbt object, use its plotting
            st.subheader("Portfolio Stats")
            st.text(pf.stats())
            st.subheader("Equity Curve")
            st.plotly_chart(pf.plot(), use_container_width=True)
        elif hasattr(pf, "equity"):
            series = pf.equity
            st.subheader("Equity Curve")
            st.line_chart(series)
        else:
            # fallback for simplified engine
            st.subheader("Equity Curve")
            st.line_chart(pf)

    # ------------------------------
    # Batch Mode
    # ------------------------------
    else:
        try:
            fasts = [int(x.strip()) for x in fast_list_raw.split(',') if x.strip().isdigit()]
            slows = [int(x.strip()) for x in slow_list_raw.split(',') if x.strip().isdigit()]
        except ValueError:
            st.error("Invalid input in Fast or Slow windows lists")
            st.stop()

        if not fasts or not slows:
            st.error("Please provide valid comma-separated integers for windows.")
            st.stop()

        # Run batch
        with st.spinner('Running batch backtest...'):
            pf = sma_crossover_backtest_batch(price, fasts, slows)
            
            st.subheader("Batch Results (Total Return %)")
            
            # Heatmap logic for VectorBT
            try:
                # Extract total return
                returns = pf.total_return()
                
                # Unstack to create a grid: Fast (index) vs Slow (columns)
                # VectorBT usually produces a MultiIndex on columns or index
                heatmap_data = returns.unstack()
                
                st.dataframe(heatmap_data.style.background_gradient(cmap='RdYlGn').format("{:.2%}"))
                
                best_idx = returns.idxmax()
                st.success(f"Best Combination: {best_idx} | Return: {returns.max():.2%}")
                
            except Exception as e:
                st.error(f"Could not display heatmap: {e}")
                st.write(pf.total_return())