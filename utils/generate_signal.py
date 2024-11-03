import pandas as pd
import numpy as np
from utils.indicators import Indicator

async def get_signal(df, tolerance, breakout_threshold, ma_periods=(10, 48), ema_period=200, bb_period=20, std_dev=2):
    """
    Check for buy/sell signals based on support/resistance levels and breakout conditions of indicators.
    Args:
        df: DataFrame with price data.
        tolerance: Tolerance percentage for considering price near the indicator.
        breakout_threshold: Threshold percentage beyond which a breakout is detected.
        ma_periods: Tuple of periods for moving averages (e.g., MA10, MA48).
        ema_period: Period for EMA (e.g., EMA200).
        bb_period: Period for Bollinger Bands.
        std_dev: Standard deviation for Bollinger Bands.
    Returns:
        str: 'buy', 'sell', 'breakout_buy', 'breakout_sell', or 'neutral' based on indicator levels.
    """
    indicator = Indicator(df)
    tolerance = min(tolerance) / 2
    
    # Calculate moving averages and EMA
    for ma_period in ma_periods:
        df[f'MA{ma_period}'] = indicator.moving_average(ma_period).dropna()
    df['EMA200'] = indicator.ema(ema_period).dropna()
    
    # Calculate Bollinger Bands
    df['BB_Mid'] = df['close'].rolling(window=bb_period).mean()
    df['BB_STD'] = df['close'].rolling(window=bb_period).std()
    df['BB_High'] = df['BB_Mid'] + (std_dev * df['BB_STD'])
    df['BB_Low'] = df['BB_Mid'] - (std_dev * df['BB_STD'])

    window_length = 20
    df['Support'] = df['low'].rolling(window=window_length).min()
    df['Resistance'] = df['high'].rolling(window=window_length).max()


    # Identify bullish breakouts (closing price greater than resistance)
    df['Bullish Breakout'] = df['close'] > df['Resistance'].shift()

    # Identify bearish breakouts (closing price less than support)
    df['Bearish Breakout'] = df['close'] < df['Support'].shift()

    # Define support/resistance and breakout checks for each indicator
    def check_level(df, indicator_val, tolerance, breakout_threshold):
        """Returns support/resistance, breakout, or None based on tolerance and breakout conditions."""
        price = df['close'].iloc[-1]
        support = df['Support'].iloc[-1]
        resistance = df['Resistance'].iloc[-1]
        

        bullish_breakout = df['Bullish Breakout'].iloc[-1]
        bearish_breakout = df['Bearish Breakout'].iloc[-1]
        #print("bullish:", bullish_breakout, "- bearish:", bearish_breakout)

        distance = abs(price - indicator_val) / indicator_val * 100
        if distance <= tolerance:
            if price < indicator_val and not (bearish_breakout):
                return 'support'
            elif price > indicator_val and not (bullish_breakout):
                return 'resistance'
        # elif distance > breakout_threshold :
        #     return 'breakout_buy' if price > indicator_val else 'breakout_sell'
        return None

    #     if (abs(price - support) / support <= tolerance) and not (bearish_breakout):
    #         return 'support'
    
    # # Check proximity to resistance
    #     elif (abs(price - resistance) / resistance <= tolerance) and not (bullish_breakout):
    #         return 'resistance'
    # Get the latest price and signals
    latest_price = df['close'].iloc[-1]
    signals = {'buy': 0, 'sell': 0, 'breakout_buy': 0, 'breakout_sell': 0}

    # Check each indicator for support/resistance or breakout
    for ma_period in ma_periods:
        level_result = check_level(df, df[f'MA{ma_period}'].iloc[-1], tolerance, breakout_threshold)
        if level_result == 'support':
            signals['buy'] += 1
        elif level_result == 'resistance':
            signals['sell'] += 1
        elif level_result == 'breakout_buy':
            signals['breakout_buy'] += 1
        elif level_result == 'breakout_sell':
            signals['breakout_sell'] += 1

    # EMA200 check
    level_result = check_level(df, df['EMA200'].iloc[-1], tolerance, breakout_threshold)
    if level_result == 'support':
        signals['buy'] += 1
    elif level_result == 'resistance':
        signals['sell'] += 1
    elif level_result == 'breakout_buy':
        signals['breakout_buy'] += 1
    elif level_result == 'breakout_sell':
        signals['breakout_sell'] += 1

    # Bollinger Bands checks
    level_result = check_level(df, df['BB_High'].iloc[-1], tolerance, breakout_threshold)
    if level_result == 'resistance':
        signals['sell'] += 1
    elif level_result == 'breakout_sell':
        signals['breakout_sell'] += 1

    level_result = check_level(df, df['BB_Low'].iloc[-1], tolerance, breakout_threshold)
    if level_result == 'support':
        signals['buy'] += 1
    elif level_result == 'breakout_buy':
        signals['breakout_buy'] += 1

    # Middle Bollinger Band handling
    level_result = check_level(df, df['BB_Mid'].iloc[-1], tolerance, breakout_threshold)
    if level_result == 'support':
        signals['buy'] += 1
    elif level_result == 'resistance':
        signals['sell'] += 1

    # Determine the final signal based on counts
    # if signals['breakout_buy'] > 0:
    #     print("breakout buy")
    #     return 'BUY'
    # elif signals['breakout_sell'] > 0:
    #     print("breakout sell")
    #     return 'SELL'
    # el
    if signals['buy'] > signals['sell']:
        return 'BUY'
    elif signals['sell'] > signals['buy']:
        return 'SELL'
    else:
        return 'HOLD'
