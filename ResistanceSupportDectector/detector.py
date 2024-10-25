import pandas as pd
import numpy as np
from utils import indicators
from utils.indicators import Indicator
import asyncio
# import pandas_ta as ta

async def is_support_resistance(df, ma_period=10):
    """
    Determines whether the moving average (MA) is acting as support, resistance, or neutral.
    Parameters:
    - df: DataFrame containing at least the 'close' price column.
    - ma_period: Length of the moving average.
    Returns:
    - 'support' if the MA is acting as support, 'resistance' if acting as resistance,
      or 'neutral' if neither.
    """
    # Calculate the Moving Average (MA)
    indicator = Indicator(df)
    df['MA'] = indicator.moving_average(ma_period).dropna()

    # Calculate the slope of the MA (using the difference between the last two MAs)
    df['MA_slope'] = df['MA'].diff()

    # Set initial state as neutral
    support_resistance = 'neutral'

    # Determine if the price is above or below the MA
    if df['close'].iloc[-1] > df['MA'].iloc[-1]:
        if df['close'].iloc[-2] < df['MA'].iloc[-2]:
            support_resistance = 'resistance'

    elif df['close'].iloc[-1] < df['MA'].iloc[-1]:
        if df['close'].iloc[-2] > df['MA'].iloc[-2]:
            support_resistance = 'support'

    # Refine by considering the slope of the MA
    if support_resistance == 'resistance' and df['MA_slope'].iloc[-1] > 0:
        support_resistance = 'neutral'
    elif support_resistance == 'support' and df['MA_slope'].iloc[-1] < 0:
        support_resistance = 'neutral'

    # Check for bounces and increase accuracy
    bounce_count = 0
    for i in range(-ma_period, -1):
        if (df['close'].iloc[i] > df['MA'].iloc[i] and df['close'].iloc[i-1] < df['MA'].iloc[i-1]) or \
           (df['close'].iloc[i] < df['MA'].iloc[i] and df['close'].iloc[i-1] > df['MA'].iloc[i-1]):
            bounce_count += 1

    if bounce_count >= 2:
        if support_resistance == 'neutral' and df['close'].iloc[-1] > df['MA'].iloc[-1]:
            support_resistance = 'resistance'
        elif support_resistance == 'neutral' and df['close'].iloc[-1] < df['MA'].iloc[-1]:
            support_resistance = 'support'

    return support_resistance

def ma_support_resistance(df, period, tolerance, min_touches=2, recent_window=3):
    """
    Determines if the Moving Average (MA) is acting as support, resistance, or neutral.
    Parameters:
    df (pd.DataFrame): DataFrame containing at least a 'close' column with price data.
    period (int): The period to calculate the moving average.
    tolerance (float): A percentage value to account for price nearing the MA without exactly touching it.
    min_touches (int): Minimum number of times the price has to touch the MA to confirm support/resistance.
    recent_window (int): How many recent candles to check for a price-MA interaction.
    Returns:
    str: 'support', 'resistance', or 'neutral' based on the relationship between prices and MA.
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column for price data.")

    df['ma'] = df['close'].rolling(window=period).mean().dropna()

    if len(df) < period + recent_window:
        raise ValueError(f"Not enough data to compute {period}-period moving average and make decisions.")

    recent_prices = df['close'].iloc[-recent_window:]
    recent_ma = df['ma'].iloc[-recent_window:]

    touches = sum(abs(recent_prices - recent_ma) / recent_ma <= tolerance)

    if all(recent_prices > recent_ma) and touches >= min_touches:
        return 'support'
    elif all(recent_prices < recent_ma) and touches >= min_touches:
        return 'resistance'

    crossings = sum((recent_prices > recent_ma).astype(int).diff().fillna(0).abs() > 0)

    if crossings >= min_touches:
        return 'neutral'

    return 'neutral'


async def is_price_near_ma(df, tolerance, ma_period):
    """
    Checks if the current price is within a tolerance of the MA and determines if the MA is acting as resistance or support.
    Args:
        df: Pandas DataFrame containing price data with columns 'close' and 'Date'.
        ma_period: Length of the moving average.
        tolerance: Percentage tolerance for considering price near MA.
        breakout_value: Percentage value for considering a breakout beyond the MA.
    Returns:
        'resistance', 'support', or None (indicating a breakout beyond the threshold).
    """
    indicator = Indicator(df)
    df['ma'] = indicator.moving_average(ma_period).dropna()

    latest_row = df.iloc[-1]
    price = latest_row['close']
    ma = latest_row['ma']
    # dist = distance_to_indicator(price, ma)
    # print("distance to moving average :", dist)
    # #print("=======================================")

    # if (price - ma) / ma > tolerance / 2:
    #     return None
    #print("ma tolerance: ", (abs(price - ma) / ma) * 100)
    # if abs(price - ma) / ma <= tolerance:
    #     if price > ma:
    #         return 'resistance'
    #     else:
    #         return 'support'
    # else:
    #     return None
    if (abs(price - ma) / ma) * 100 <= tolerance:
        if price > ma:
            return "BUY"
        else:
            return "SELL"
    else:
        return None


async def check_ema(df, tolerance, period=200):
    """
    Check if price is near EMA and if it's acting as support or resistance. If price breaks out beyond a threshold, signal will be False.
    Args:
        df (pd.DataFrame): DataFrame containing price data.
        tolerance (float): Tolerance for price proximity to EMA.
        breakout_value (float): Threshold for breakout beyond the EMA.
        period (int, optional): Period for EMA calculation. Defaults to 200.
    """
    indicator = Indicator(df)
    df['ema'] = indicator.ema(period).dropna()

    latest_row = df.iloc[-1]
    price = latest_row['close']
    ema = latest_row['ema']
    # dist = distance_to_indicator(price, ema)
    # print("distance to ema :", dist)
    # #print("=======================================")
    #print("ema tolerance: ", (abs(price - ema) / ema) * 100)
    # if (price - ema) / ema > tolerance / 2:
    #     return None

    if (abs(price - ema) / ema) * 100 <= tolerance:
        if price > ema:
            return "BUY"
        else:
            return "SELL"
    else:
        return None


# async def is_bollinger_band_support_resistance(df, period=20, std_dev=2):
#     """
#     Determines whether the Bollinger Band is acting as support or resistance.
#     Args:
#         df: Pandas DataFrame containing price data with columns 'close'.
#         period: Period for calculating the moving average.
#         std_dev: Number of standard deviations for the Bollinger Bands.
#     Returns:
#         'support', 'resistance', or 'neutral'.
#     """
#     bbands = ta.bbands(df['close'], length=20, std=2)
#     if bbands is not None:
#         df[['BB_Low','BB_Mid', 'BB_High']] = bbands.iloc[:, :3]
#     else:
#         raise ValueError("Failed to compute Bollinger Bands.")

#     if df['close'].iloc[-1] <= df['BB_Low'].iloc[-1] and df['close'].iloc[-2] > df['BB_Low'].iloc[-2]:
#         return 'support'
#     elif df['close'].iloc[-1] >= df['BB_High'].iloc[-1] and df['close'].iloc[-2] < df['BB_High'].iloc[-2]:
#         return 'resistance'
#     else:
#         return 'neutral'


# async def is_price_near_bollinger_band(df, tolerance, period=20, std_dev=2):
#     """
#     Checks if the current price is near the upper or lower Bollinger Band.
#     Args:
#         df: Pandas DataFrame containing price data with columns 'close'.
#         period: Period for calculating the moving average.
#         std_dev: Number of standard deviations for the Bollinger Bands.
#         tolerance: Percentage tolerance for considering price near the band.
#     Returns:
#         'upper_band', 'lower_band', or 'neutral'.
#     """
#     bbands = ta.bbands(df['close'], length=20, std=2)
#     if bbands is not None:
#         df[['BB_Low','BB_Mid', 'BB_High']] = bbands.iloc[:, :3]
#     else:
#         raise ValueError("Failed to compute Bollinger Bands.")


#     last_price = df['close'].iloc[-1]
#     upper_band_value = df['BB_High'].iloc[-1]
#     lower_band_value = df['BB_Low'].iloc[-1]

#     # dist = distance_to_indicator(last_price, upper_band_value)
#     # print("distance to upper band :", dist)

#     # low_dist = distance_to_indicator(last_price, lower_band_value)
#     # print("distance to lower band :", low_dist)
#     # #print("=======================================")

#     if abs(last_price - upper_band_value) <= tolerance * upper_band_value:
#         return 'upper_band'
#     elif abs(last_price - lower_band_value) <= tolerance * lower_band_value:
#         return 'lower_band'
#     else:
#         return 'neutral'

# def distance_to_indicator(current_price, indicator_value):
#     """
#     Calculates the percentage distance between the current market price and an indicator.
#     :param current_price: The current price of the market (float).
#     :param indicator_value: The value of the technical indicator (float).
#     :return: The percentage distance (float).
#     """
#     # Calculate absolute difference between current price and indicator
#     difference = abs(current_price - indicator_value)

#     # Calculate the percentage distance (relative to current price)
#     percentage_distance = (difference / current_price) * 100
    
#     return percentage_distance

import pandas as pd

async def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    Calculates Bollinger Bands for the given DataFrame.
    Args:
        df: Pandas DataFrame containing price data with columns 'close'.
        period: Period for calculating the moving average.
        std_dev: Number of standard deviations for the Bollinger Bands.
    Returns:
        DataFrame with Bollinger Bands (BB_Low, BB_Mid, BB_High).
    """
    df['BB_Mid'] = df['close'].rolling(window=period).mean()
    df['BB_STD'] = df['close'].rolling(window=period).std()
    df['BB_High'] = df['BB_Mid'] + (std_dev * df['BB_STD'])
    df['BB_Low'] = df['BB_Mid'] - (std_dev * df['BB_STD'])

    return df[['BB_Low', 'BB_Mid', 'BB_High']]

async def is_bollinger_band_support_resistance(df, period=20, std_dev=2):
    """
    Determines whether the Bollinger Band is acting as support or resistance.
    Args:
        df: Pandas DataFrame containing price data with columns 'close'.
        period: Period for calculating the moving average.
        std_dev: Number of standard deviations for the Bollinger Bands.
    Returns:
        'support', 'resistance', or 'neutral'.
    """
    # Calculate Bollinger Bands
    df[['BB_Low', 'BB_Mid', 'BB_High']] = await calculate_bollinger_bands(df, period, std_dev)

    if df['close'].iloc[-1] <= df['BB_Low'].iloc[-1] and df['close'].iloc[-2] > df['BB_Low'].iloc[-2]:
        return 'support'
    elif df['close'].iloc[-1] >= df['BB_High'].iloc[-1] and df['close'].iloc[-2] < df['BB_High'].iloc[-2]:
        return 'resistance'
    else:
        return 'neutral'

async def is_price_near_bollinger_band(df, high_tol, low_tol, period=20, std_dev=2):
    """
    Checks if the current price is near the upper or lower Bollinger Band.
    Args:
        df: Pandas DataFrame containing price data with columns 'close'.
        period: Period for calculating the moving average.
        std_dev: Number of standard deviations for the Bollinger Bands.
        tolerance: Percentage tolerance for considering price near the band.
    Returns:
        'upper_band', 'lower_band', or 'neutral'.
    """
    # Calculate Bollinger Bands
    df[['BB_Low', 'BB_Mid', 'BB_High']] = await calculate_bollinger_bands(df, period, std_dev)

    last_price = df['close'].iloc[-1]
    upper_band_value = df['BB_High'].iloc[-1]
    lower_band_value = df['BB_Low'].iloc[-1]
    #print("upper band tolerance: ", (abs(last_price - upper_band_value) / upper_band_value) * 100)
    #print("lower band tolerance: ", (abs(last_price - lower_band_value) / lower_band_value) * 100)
    if (last_price - upper_band_value) / upper_band_value > high_tol / 2 or (lower_band_value - last_price) / lower_band_value > low_tol / 2:
        return "neutral"
    
    upper_tolerance = (abs(last_price - upper_band_value) / upper_band_value) * 100
    lower_tolerance = (abs(last_price - lower_band_value) / lower_band_value) * 100
    #if upper_tolerance < lower_tolerance:
    if upper_tolerance <= high_tol or lower_tolerance <= low_tol:
        if last_price > upper_band_value or last_price < lower_band_value:
            return "BUY"
        else:
            return "SELL"
    else:
        return None
    #     return 'upper_band'
    # elif abs(last_price - lower_band_value) <= low_tol * lower_band_value:
    #     return 'lower_band'
    # else:
    #     return 'neutral'

def distance_to_indicator(current_price, indicator_value):
    """
    Calculates the percentage distance between the current market price and an indicator.
    Args:
        current_price: The current price of the market (float).
        indicator_value: The value of the technical indicator (float).
    Returns:
        The percentage distance (float).
    """
    # Calculate absolute difference between current price and indicator
    difference = abs(current_price - indicator_value)

    # Calculate the percentage distance (relative to current price)
    percentage_distance = (difference / current_price) * 100
    
    return percentage_distance
