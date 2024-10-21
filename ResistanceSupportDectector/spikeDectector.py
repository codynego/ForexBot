import pandas as pd

def detect_spikes(df, threshold=0.02, lookback=5):
    """
    Detects spikes in price data based on percentage changes and recent volatility.

    Args:
        df: Pandas DataFrame containing price data with 'Close' and 'Date' columns.
        threshold: Percentage threshold for detecting spikes (e.g., 2%).
        lookback: Number of periods to look back for calculating average price.

    Returns:
        A string indicating the type of spike detected: "spike_up", "spike_down", or "no_spike".
    """
    # Calculate percentage changes
    df['pct_change'] = df['close'].pct_change()
    
    # Calculate recent average price for comparison
    recent_avg_price = df['close'].rolling(window=lookback).mean().shift(1)

    # Check for spikes
    last_pct_change = df['pct_change'].iloc[-1]
    last_price = df['close'].iloc[-1]

    # Determine spike conditions
    if last_pct_change > threshold:
        return "spike_up"
    elif last_pct_change < -threshold:
        return "spike_down"
    else:
        return "no_spike"
