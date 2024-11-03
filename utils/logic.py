from ResistanceSupportDectector.detector import calculate_bollinger_bands
from utils.indicators import Indicator
import pandas as pd
import numpy as np
from utils.indicators import Indicator

async def get_signal(df, tolerance, ma_periods=(10, 48), ema_period=200, bb_period=20, std_dev=2):
    """
    Check for buy/sell signals based on support and resistance levels of indicators across multiple timeframes.
    Args:
        df: DataFrame with price data.
        tolerance: Tolerance percentage for considering price near the indicator.
        ma_periods: Tuple of periods for moving averages (e.g., MA10, MA48).
        ema_period: Period for EMA (e.g., EMA200).
        bb_period: Period for Bollinger Bands.
        std_dev: Standard deviation for Bollinger Bands.
    Returns:
        str: 'buy', 'sell', or 'neutral' based on indicator levels.
    """
    indicator = Indicator(df)
    
    # Calculate moving averages and EMA
    for ma_period in ma_periods:
        df[f'MA{ma_period}'] = indicator.moving_average(ma_period).dropna()
    df['EMA200'] = indicator.ema(ema_period).dropna()
    
    # Calculate Bollinger Bands
    df['BB_Mid'] = df['close'].rolling(window=bb_period).mean()
    df['BB_STD'] = df['close'].rolling(window=bb_period).std()
    df['BB_High'] = df['BB_Mid'] + (std_dev * df['BB_STD'])
    df['BB_Low'] = df['BB_Mid'] - (std_dev * df['BB_STD'])

    # Define support/resistance checks for each indicator
    def check_level(price, indicator_val):
        """Return True if price is near the indicator within the given tolerance."""
        return abs(price - indicator_val) / indicator_val <= tolerance

    # Get the latest price and signals
    latest_price = df['close'].iloc[-1]
    signals = {'buy': 0, 'sell': 0}

    # Check each indicator for support/resistance
    for ma_period in ma_periods:
        if check_level(latest_price, df[f'MA{ma_period}'].iloc[-1]):
            if latest_price > df[f'MA{ma_period}'].iloc[-1]:
                signals['sell'] += 1
            else:
                signals['buy'] += 1

    if check_level(latest_price, df['EMA200'].iloc[-1]):
        if latest_price > df['EMA200'].iloc[-1]:
            signals['sell'] += 1
        else:
            signals['buy'] += 1

    if check_level(latest_price, df['BB_High'].iloc[-1]):
        signals['sell'] += 1
    elif check_level(latest_price, df['BB_Low'].iloc[-1]):
        signals['buy'] += 1
    elif check_level(latest_price, df['BB_Mid'].iloc[-1]):
        # Decide how to handle the middle Bollinger Band (e.g., neutral or trend-following)
        if latest_price > df['BB_Mid'].iloc[-1]:
            signals['buy'] += 1
        else:
            signals['sell'] += 1

    # Determine the final signal based on counts
    if signals['buy'] > signals['sell']:
        return 'buy'
    elif signals['sell'] > signals['buy']:
        return 'sell'
    else:
        return 'neutral'



async def is_price_near_ma_with_breakout(df, tolerance, breakout_threshold, ma_period):
    """
    Checks if the current price is within a tolerance of the MA or if it has broken out.
    Args:
        df: Pandas DataFrame containing price data with columns 'close' and 'Date'.
        tolerance: Percentage tolerance for considering price near MA.
        breakout_threshold: Percentage distance from MA to consider as a breakout.
        ma_period: Length of the moving average.
    Returns:
        'resistance', 'support', 'breakout_buy', 'breakout_sell', or None.
    """
    indicator = Indicator(df)
    df['ma'] = indicator.moving_average(ma_period).dropna()

    latest_row = df.iloc[-1]
    price = latest_row['close']
    ma = latest_row['ma']

    # Calculate distance between price and MA
    distance_to_ma = abs(price - ma) / ma * 100

    # Determine if the price is near MA within tolerance
    if distance_to_ma <= tolerance:
        if price > ma:
            return "resistance"
        elif price < ma:
            return "support"
    
    # Breakout conditions
    elif distance_to_ma > breakout_threshold:
        if price > ma:
            return "breakout_buy"
        elif price < ma:
            return "breakout_sell"

    return None


async def is_price_near_bollinger_with_breakout(df, high_tol, low_tol, breakout_threshold, period=20, std_dev=2):
    """
    Checks if the current price is near the Bollinger Bands or has broken out.
    Args:
        df: Pandas DataFrame containing price data with columns 'close'.
        high_tol: Upper tolerance for considering price near the upper band.
        low_tol: Lower tolerance for considering price near the lower band.
        breakout_threshold: Threshold beyond which a breakout is considered.
    Returns:
        'upper_band', 'lower_band', 'breakout_buy', 'breakout_sell', or 'neutral'.
    """
    df[['BB_Low', 'BB_Mid', 'BB_High']] = await calculate_bollinger_bands(df, period, std_dev)

    last_price = df['close'].iloc[-1]
    upper_band_value = df['BB_High'].iloc[-1]
    lower_band_value = df['BB_Low'].iloc[-1]

    # Calculate distances for breakout check
    upper_distance = (last_price - upper_band_value) / upper_band_value * 100
    lower_distance = (lower_band_value - last_price) / lower_band_value * 100

    # Breakout check
    if upper_distance > breakout_threshold:
        return "breakout_buy"
    elif lower_distance > breakout_threshold:
        return "breakout_sell"

    # Check if near upper or lower band within tolerance
    if abs(last_price - upper_band_value) / upper_band_value * 100 <= high_tol:
        return "upper_band"
    elif abs(last_price - lower_band_value) / lower_band_value * 100 <= low_tol:
        return "lower_band"

    return "neutral"
