# Local Financial AI Agent (Local Ollama LLM + nomic-embed-text) — with Backtesting & Streamlit

This repository now includes:
- VectorBT-based backtester (vectorized, fast)
- Backtrader-based strategy example (event-driven)
- Streamlit dashboard to fetch yahoo ticker data and call the Decision Agent

## Quickstart

1. Create and activate a virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Ensure Ollama is running locally and `nomic-embed-text:latest` is pulled, plus your preferred LLM model.

3. Populate `data/` with `trade_history.csv`, `portfolio.json`, and `config/strategy_rules.txt`.

4. Build embeddings and FAISS index:
   ```python
   # minimal example
   from rag.rag_engine import RAGEngine
   r = RAGEngine()
   r.add_document(open('config/strategy_rules.txt').read())
   ```

5. Run the dashboard:
   ```bash
   streamlit run ui/streamlit_app.py --server.port 8888
   streamlit run ui/trading_signals_dashboard.py --server.port=8888
   ```

6. Or run main:
   ```bash
   python main.py
   ```

## Notes
- vectorbt may have additional optional dependencies; consult their docs if install issues occur.
- Backtesting is for research and must be validated before any live trading.
