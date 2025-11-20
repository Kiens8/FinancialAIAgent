
import pandas as pd

def generate_sma_signals(price, fast=10, slow=20):
    df = pd.DataFrame(price)
    df['fast'] = df.iloc[:,0].rolling(fast).mean()
    df['slow'] = df.iloc[:,0].rolling(slow).mean()
    df['signal'] = (df['fast'] > df['slow']).astype(int)
    return df
