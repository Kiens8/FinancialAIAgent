import backtrader as bt
import pandas as pd

class SmaCross(bt.SignalStrategy):
    def __init__(self):
        sma1 = bt.ind.SMA(self.data.close, period=10)
        sma2 = bt.ind.SMA(self.data.close, period=50)
        self.signal_add(bt.SIGNAL_LONG, sma1 > sma2)

def run_backtrader(price_df: pd.DataFrame, cash=10000):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=price_df)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0005)
    results = cerebro.run()
    final_value = cerebro.broker.getvalue()
    return final_value
