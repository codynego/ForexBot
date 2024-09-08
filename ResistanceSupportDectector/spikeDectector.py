import pandas as pd
import numpy as np

def detect_spikes(df, spike_threshold=2):
  """
  Detects price spikes in a DataFrame.

  Args:
    df: Pandas DataFrame with a 'close' column.
    spike_threshold: Number of standard deviations for a spike.

  Returns:
    A DataFrame with a 'spike' column indicating whether a spike occurred.
  """

  df['price_change'] = df['close'].diff()
  df['std'] = df['price_change'].rolling(window=20).std()
  df['spike'] = abs(df['price_change']) > spike_threshold * df['std']

  return df[df['spike'] == True]