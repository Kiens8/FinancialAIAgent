import json
import os
import yfinance as yf
from agents.strategy_research_agent import StrategyResearchAgent
from agents.decision_agent import DecisionAgent
from signals.signal_engine import generate_signals
from utils.file import load_text
CONFIG_DIR = "config"
FAV_FILE = os.path.join(CONFIG_DIR, "favorites.json")
DEFAULT_FAVS = ["AAPL", "MSFT", "TSLA"]

# -----------------------------
# Load favorites.json
# -----------------------------
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


# -----------------------------
# Fetch OHLCV data — IMPORTANT:
# Must include High, Low, Close to match signal_engine
# -----------------------------
def fetch_price_data(ticker, days=90):
    try:
        df = yf.download(
            ticker,
            period=f"{days}d",
            interval="1d",
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            print(f"{ticker}: No yfinance data.")
            return None

        # Must keep High, Low, Close for ATR
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df

    except Exception as e:
        print(f"Error loading {ticker}: {e}")
        return None


# -----------------------------
# Ask local LLM to validate indicator signal
# -----------------------------
# -----------------------------
# Ask local LLM to validate indicator signal
# -----------------------------
def ask_llm_validation(decision_agent, ticker, signal, confidence, details):
    """
    Use the existing DecisionAgent.decide() method instead of non-existent .explain().
    """
    market_state = {
        "ticker": ticker,
        "signal": signal,
        "confidence": confidence,
        "details": details
    }

    # Call decide() instead of explain()
    raw_response = decision_agent.decide(market_state)

    # Parse LLM response to extract one-word signal
    llm_text = ""
    try:
        # Attempt to extract from JSON response
        if isinstance(raw_response, str):
            import json
            try:
                parsed = json.loads(raw_response)
                llm_text = parsed.get("signal") or parsed.get("decision") or ""
            except Exception:
                llm_text = raw_response
        elif isinstance(raw_response, dict):
            llm_text = raw_response.get("signal") or raw_response.get("decision") or ""
        else:
            llm_text = str(raw_response)
    except Exception:
        llm_text = str(raw_response)

    llm_text = llm_text.strip()

    # Extract final action: Buy, Sell, Hold
    for w in ["buy", "sell", "hold"]:
        if w in llm_text.lower():
            return w.capitalize()
    return "Hold"

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == '__main__':

    print("Loading agents...")
    research_agent = StrategyResearchAgent()
    decision_agent = DecisionAgent()

    # Load rules
    rules = load_text("config/strategy_rules.txt")
    if rules:
        research_agent.rag.add_document(rules)

    print("\nReading favorites.json...")
    favorites = load_favorites()

    if not favorites:
        print("No favorites found. Exiting.")
        exit()

    print("\n=== Running signal validation for favorite stocks ===\n")

    for ticker in favorites:
        print(f"Processing {ticker}...")

        df = fetch_price_data(ticker)
        if df is None:
            print(f"Skipping {ticker}, no data.\n")
            continue

        # ---------------------------------------
        # 🔥 Use SAME indicator engine as UI
        # ---------------------------------------
        result = generate_signals(df)

        indicator_signal = result["signal"]
        confidence = result["confidence"]
        details = result["details"]

        # ---------------------------------------
        # 🔥 Send signal to LLM for validation
        # ---------------------------------------
        llm_signal = ask_llm_validation(
            decision_agent,
            ticker,
            indicator_signal,
            confidence,
            details
        )

        # Compare results
        conclusion = "Match" if llm_signal.upper() == indicator_signal.upper() else "Not Match"

        print(f"""
-----------------------------------------
Ticker: {ticker}
Indicator Signal : {indicator_signal}
Signal LLM       : {llm_signal}
Conclusion       : {conclusion}
-----------------------------------------
""")
