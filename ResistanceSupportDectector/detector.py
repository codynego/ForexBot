import pandas as pd
import numpy as np
from utils import indicators
from utils.indicators import Indicator
import asyncio

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
    #df['MA'] = df['close'].rolling(window=ma_period).mean()
    df['MA'] = indicator.moving_average(ma_period)

    # Calculate the slope of the MA (using the difference between the last two MAs)
    df['MA_slope'] = df['MA'].diff()

    # Set initial state as neutral
    support_resistance = 'neutral'

    # Determine if the price is above or below the MA
    if df['close'].iloc[-1] > df['MA'].iloc[-1]:
        # Price is above the MA, possible resistance
        if df['close'].iloc[-2] < df['MA'].iloc[-2]:
            support_resistance = 'resistance'

    elif df['close'].iloc[-1] < df['MA'].iloc[-1]:
        # Price is below the MA, possible support
        if df['close'].iloc[-2] > df['MA'].iloc[-2]:
            support_resistance = 'support'

    # Refine the determination by considering the slope of the MA
    if support_resistance == 'resistance' and df['MA_slope'].iloc[-1] > 0:
        # Price is above MA, but MA is sloping upward (indicating an uptrend)
        # More likely to be support in an uptrend
        support_resistance = 'neutral'

    elif support_resistance == 'support' and df['MA_slope'].iloc[-1] < 0:
        # Price is below MA, but MA is sloping downward (indicating a downtrend)
        # More likely to be resistance in a downtrend
        support_resistance = 'neutral'

    # Further refine by checking for bounces
    # Count the number of bounces off the MA in recent periods
    bounce_count = 0
    for i in range(-ma_period, -1):
        if (df['close'].iloc[i] > df['MA'].iloc[i] and df['close'].iloc[i-1] < df['MA'].iloc[i-1]) or \
           (df['close'].iloc[i] < df['MA'].iloc[i] and df['close'].iloc[i-1] > df['MA'].iloc[i-1]):
            bounce_count += 1

    # If the bounce count is high, it suggests a strong support/resistance level
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
    # Ensure the DataFrame has the 'close' column
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column for price data.")

    # Calculate the moving average
    df['ma'] = df['close'].rolling(window=period).mean()

    # Ensure enough data for comparison
    if len(df) < period + recent_window:
        raise ValueError(f"Not enough data to compute {period}-period moving average and make decisions.")

    # Get recent price and MA data over the recent_window
    recent_prices = df['close'].iloc[-recent_window:]
    recent_ma = df['ma'].iloc[-recent_window:]

    # Count how many times the price has touched or gotten close to the MA
    touches = sum(abs(recent_prices - recent_ma) / recent_ma <= tolerance)

    # Check for support condition
    if all(recent_prices > recent_ma) and touches >= min_touches:
        return 'support'

    # Check for resistance condition
    elif all(recent_prices < recent_ma) and touches >= min_touches:
        return 'resistance'

    # Check for neutral if prices frequently cross above and below the MA
    crossings = sum((recent_prices > recent_ma).astype(int).diff().fillna(0).abs() > 0)

    if crossings >= min_touches:
        return 'neutral'

    return 'neutral'



# async def is_price_near_ma(df, tolerance, ma_period):
#   """
#   Checks if the current price is within a tolerance of the MA.

  # Args:
  #   df: Pandas DataFrame containing price data with columns 'close' and 'Date'.
  #   ma_period: Length of the moving average.
  #   tolerance: Percentage tolerance for considering price near MA.

  # Returns:
  #   True if the price is within tolerance of the MA, False otherwise.
  # """

  # indicator = Indicator(df)
  #   #df['MA'] = df['close'].rolling(window=ma_period).mean()
  # df['MA'] = indicator.moving_average(ma_period)
  # last_price = df['close'].iloc[-1]
  # last_ma = df['MA'].iloc[-1]
  # return abs(last_price - last_ma) / last_ma <= tolerance

async def is_price_near_ma(df, tolerance, ma_period):
    """
    Checks if the current price is within a tolerance of the MA and determines if the MA is acting as resistance or support.

    Args:
        df: Pandas DataFrame containing price data with columns 'close' and 'Date'.
        ma_period: Length of the moving average.
        tolerance: Percentage tolerance for considering price near MA.

    Returns:
        'resistance' if the MA is acting as resistance,
        'support' if the MA is acting as support,
        None if the price is not near the MA.
    """

    indicator = Indicator(df)
    df['ma'] = indicator.moving_average(ma_period)
  
    
    # Get the latest row in the DataFrame
    latest_row = df.iloc[-1]
    price = latest_row['close']  # Assuming 'close' is the column for closing prices
    ma = latest_row['ma']
    print("ma: ", ma, "price:", price)
    
    if abs(price - ma) / ma <= tolerance:
        if price > ma:
            return 'resistance'
        else:
            return 'support'
    else:
        return None


async def check_ema(df, tolerance, period=200):
    """Check if price is near EMA and if it's acting as support or resistance.

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        period (int, optional): Period for EMA calculation. Defaults to 200.
        tolerance (float, optional): Tolerance for price proximity to EMA. Defaults to 0.001.
    """
    indicator = Indicator(df)
    df['ema'] = indicator.ema(period)
    
    # Get the latest row in the DataFrame
    latest_row = df.iloc[-1]
    price = latest_row['close']  # Assuming 'close' is the column for closing prices
    ema = latest_row['ema']
    print("ema: ", ema, "price:", price,  "tolerance:", price * tolerance)

    if abs(price - ema) / ema <= tolerance:
        if price > ema:
            return 'resistance'
        else:
            return 'support'
    else:
        return None  




async def is_bollinger_band_support_resistance(df, period=20, std_dev=2):
  """
  Determines whether the Bollinger Band is acting as support or resistance.

  Args:
    df: Pandas DataFrame containing price data with columns 'close'.
    period: Period for calculating the moving average.
    std_dev: Number of standard deviations for the Bollinger Bands.

  Returns:
    'support' if lower band acts as support, 'resistance' if upper band acts as resistance,
    'neutral' if neither.
  """

  # Calculate Bollinger Bands
  df['MA'] = df['close'].rolling(window=period).mean()
  df['std'] = df['close'].rolling(window=period).std()
  df['upper_band'] = df['MA'] + (df['std'] * std_dev)
  df['lower_band'] = df['MA'] - (df['std'] * std_dev)



  # Determine support or resistance
  if df['close'].iloc[-1] <= df['lower_band'].iloc[-1] and df['close'].iloc[-2] > df['lower_band'].iloc[-2]:
    return 'support'
  elif df['close'].iloc[-1] >= df['upper_band'].iloc[-1] and df['close'].iloc[-2] < df['upper_band'].iloc[-2]:
    return 'resistance'
  else:
    return 'neutral'
  


async def is_price_near_bollinger_band(df, tolerance, period=20, std_dev=2):
  """
  Checks if the current price is near the upper or lower Bollinger Band.

  Args:
    df: Pandas DataFrame containing price data with columns 'close'.
    period: Period for calculating the moving average.
    std_dev: Number of standard deviations for the Bollinger Bands.
    tolerance: Percentage tolerance for considering price near the band.

  Returns:
    'upper_band' if the price is near the upper band, 'lower_band' if near the lower band,
    'neutral' otherwise.
  """

  # Calculate Bollinger Bands
  df['MA'] = df['close'].rolling(window=period).mean()
  df['std'] = df['close'].rolling(window=period).std()
  df['upper_band'] = df['MA'] + (df['std'] * std_dev)
  df['lower_band'] = df['MA'] - (df['std'] * std_dev)

  # Check if price is near upper or lower band
  last_price = df['close'].iloc[-1]
  upper_band_value = df['upper_band'].iloc[-1]
  lower_band_value = df['lower_band'].iloc[-1]
  print("upper_band_value: ", upper_band_value, "price:", last_price, "tolerance:", last_price * tolerance)
  print("lower_band_value: ", lower_band_value, "price:", last_price, "tolerance:", last_price * tolerance)

  if abs(last_price - upper_band_value) <= tolerance * upper_band_value:
    return 'upper_band'
  elif abs(last_price - lower_band_value) <= tolerance * lower_band_value:
    return 'lower_band'
  else:
    return 'neutral'

