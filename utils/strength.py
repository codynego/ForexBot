import pandas as pd
import asyncio

async def compute_market_strength(df, tolerance):
    """
    Compute market strength based on multiple strategies that could lead to buy or sell signals.
    
    Args:
        df: DataFrame with market data (including necessary indicators).
        tolerance: Tolerance percentage for considering price near indicators.
        
    Returns:
        float: Market strength value between 0 and 1.
    """

    # Calculate Indicators
    window_length = 20
    df['Support'] = df['low'].rolling(window=window_length).min()
    df['Resistance'] = df['high'].rolling(window=window_length).max()
    
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA48'] = df['close'].rolling(window=48).mean()
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    df['BB_Mid'] = df['close'].rolling(window=20).mean()
    df['BB_STD'] = df['close'].rolling(window=20).std()
    df['BB_High'] = df['BB_Mid'] + 2 * df['BB_STD']
    df['BB_Low'] = df['BB_Mid'] - 2 * df['BB_STD']
    
    df['RSI'] = compute_rsi(df['close'], tolerance,)
    df['Stochastic_K'] = compute_stochastic(df,tolerance,)
    df['MACD'], df['MACD_Signal'] = compute_macd(df, tolerance)

    # Initialize total strength
    total_strength = 0
    total_signals = 0

    # Define strategies to calculate individual signals
    strategies = [
        strategy_support_resistance,
        strategy_bollinger_bands,
        strategy_moving_average_cross,
        strategy_ema_cross,
        strategy_rsi,
        strategy_stochastic,
        strategy_macd
    ]

    for strategy in strategies:
        signal_strength = strategy(df, tolerance)
        if signal_strength is not None:
            total_strength += signal_strength
            total_signals += 1

    # Calculate average strength if any signals were counted
    if total_signals > 0:
        market_strength = total_strength / total_signals
    else:
        market_strength = 0.5  # Default neutral strength if no strategies returned signals

    return min(max(market_strength, 0), 1)  # Ensure value is between 0 and 1

# Indicator Calculation Functions
def compute_rsi(series,tolerance, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_stochastic(df, tolerance, k_period=14):
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    stochastic_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    return stochastic_k

def compute_macd(df, tolerance, short_window=12, long_window=26, signal_window=9):
    ema_short = df['close'].ewm(span=short_window, adjust=False).mean()
    ema_long = df['close'].ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Strategy Functions
def strategy_support_resistance(df, tolerance):
    price = df['close'].iloc[-1]
    support = df['Support'].iloc[-1]
    resistance = df['Resistance'].iloc[-1]

    distance_to_support = abs(price - support) / support
    distance_to_resistance = abs(price - resistance) / resistance

    if distance_to_support <= tolerance:
        return 1  # Strong buy signal
    elif distance_to_resistance <= tolerance:
        return 0  # Strong sell signal
    return None  # Neutral signal

def strategy_bollinger_bands(df, tolerance):
    price = df['close'].iloc[-1]
    upper_band = df['BB_High'].iloc[-1]
    lower_band = df['BB_Low'].iloc[-1]

    if price > upper_band:
        return 0  # Indicates potential sell
    elif price < lower_band:
        return 1  # Indicates potential buy
    return None  # Neutral

def strategy_moving_average_cross(df, tolerance):
    ma_short = df['MA10'].iloc[-1]
    ma_long = df['MA48'].iloc[-1]

    if ma_short > ma_long:
        return 1  # Indicates bullish signal
    elif ma_short < ma_long:
        return 0  # Indicates bearish signal
    return None  # Neutral signal

def strategy_ema_cross(df, tolerance):
    ema_short = df['EMA10'].iloc[-1]
    ema_long = df['EMA200'].iloc[-1]

    if ema_short > ema_long:
        return 1  # Bullish
    elif ema_short < ema_long:
        return 0  # Bearish
    return None  # Neutral

def strategy_rsi(df, tolerance, rsi_threshold=70):
    rsi = df['RSI'].iloc[-1]

    if rsi < 30:
        return 1  # Strong buy signal
    elif rsi > rsi_threshold:
        return 0  # Strong sell signal
    return None  # Neutral

def strategy_stochastic(df, tolerance,overbought=80, oversold=20):
    stochastic_k = df['Stochastic_K'].iloc[-1]

    if stochastic_k < oversold:
        return 1  # Indicates buy
    elif stochastic_k > overbought:
        return 0  # Indicates sell
    return None  # Neutral

def strategy_macd(df, tolerance):
    macd = df['MACD'].iloc[-1]
    signal_line = df['MACD_Signal'].iloc[-1]

    if macd > signal_line:
        return 1  # Indicates buy
    elif macd < signal_line:
        return 0  # Indicates sell
    return None  # Neutral

# Usage
async def compute(df, tolerance):
    market_strength = await compute_market_strength(df, tolerance)
    return market_strength