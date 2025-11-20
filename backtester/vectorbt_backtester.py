import vectorbt as vbt
import pandas as pd
from typing import List, Tuple

def sma_crossover_backtest_single(price: pd.Series, fast: int, slow: int, fees: float=0.0005):
    """Run a single SMA crossover backtest and return vectorbt Portfolio."""
    fast = int(fast)
    slow = int(slow)

    fast_ma = vbt.MA.run(price, window=fast)
    slow_ma = vbt.MA.run(price, window=slow)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(price, entries, exits, fees=fees, slippage=fees)
    return pf


def sma_crossover_backtest_batch(price: pd.Series, params: List[Tuple[int,int]], symbol: str="SYM", fees: float=0.0005):
    """Run multiple SMA backtests and return flattened DataFrame of equity curves."""
    results = {}

    for fast, slow in params:
        pf = sma_crossover_backtest_single(price, fast, slow, fees=fees)
        val = pf.value()

        # Handle vectorbt sometimes returning DataFrame
        if isinstance(val, pd.DataFrame):
            series = val.iloc[:, 0].copy()
        else:
            series = val.copy()

        colname = f"{symbol}_{int(fast)}_{int(slow)}"
        series.name = colname
        results[colname] = series

    df = pd.concat(results.values(), axis=1)
    df.columns = list(results.keys())
    return df
