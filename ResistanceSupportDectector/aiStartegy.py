import pandas as pd
from ResistanceSupportDectector.detector import is_price_near_bollinger_band, is_bollinger_band_support_resistance, is_support_resistance
from utils.indicators import Indicator
from config import Config



def detect_spike(df, window=20, threshold=1.5):
    """
    Detects spikes in the price data based on the size of the price movement compared to recent history.

    Args:
    - df: DataFrame containing price data with a 'close' column.
    - window: Number of periods to calculate the average volatility.
    - threshold: Multiplier to determine if the current price movement is a spike.

    Returns:
    - A list of indices where spikes were detected.
    """
    spikes = []
    
    # Calculate the recent volatility
    df['price_change'] = df['close'].diff()
    df['abs_change'] = df['price_change'].abs()
    df['rolling_volatility'] = df['abs_change'].rolling(window=window).mean()

    # Detect spikes
    for i in range(window, len(df)):
        current_change = df['price_change'].iloc[i]
        rolling_volatility = df['rolling_volatility'].iloc[i]
        if abs(current_change) > threshold * rolling_volatility:
            spikes.append(i)
    
    return spikes


def calculate_pivot_points(df):
    """
    Calculates pivot points, support, and resistance levels.

    Args:
    - df: DataFrame containing 'high', 'low', 'close' prices.

    Returns:
    - A DataFrame with pivot points, support, and resistance levels.
    """

    # Calculate Pivot Point (PP), Support, and Resistance levels
    pivot_point = (df['high'] + df['low'] + df['close']) / 3
    support1 = (2 * pivot_point) - df['high']
    resistance1 = (2 * pivot_point) - df['low']
    support2 = pivot_point - (df['high'] - df['low'])
    resistance2 = pivot_point + (df['high'] - df['low'])
    support3 = df['low'] - 2 * (df['high'] - pivot_point)
    resistance3 = df['high'] + 2 * (pivot_point - df['low'])

    # Create a new DataFrame for storing pivot points, support, and resistance levels
    pivot_points_df = pd.DataFrame({
        'pivot_point': pivot_point,
        'support1': support1,
        'resistance1': resistance1,
        'support2': support2,
        'resistance2': resistance2,
        'support3': support3,
        'resistance3': resistance3
    })

    return pivot_points_df


def get_pivot_point_data(df, current_price, tolerance=0.005):
    """
    Determines if the current price is near pivot points, support, or resistance levels.

    Args:
    - df: DataFrame containing 'high', 'low', 'close' prices.
    - current_price: The current price of the asset.
    - tolerance: The allowed deviation from the pivot/support/resistance levels (default: 0.5%).

    Returns:
    - A dictionary indicating if the current price is near support or resistance levels.
    """

    # Calculate pivot points and levels
    pivot_points_df = calculate_pivot_points(df)

    # Get the last row of pivot points (for the most recent data)
    latest_pivot = pivot_points_df.iloc[-1]

    # Determine if the current price is near any key levels
    near_support = False
    near_resistance = False

    # Check if the current price is near any of the supports or resistances
    if abs(current_price - latest_pivot['support1']) / latest_pivot['support1'] <= tolerance:
        near_support = True
    elif abs(current_price - latest_pivot['support2']) / latest_pivot['support2'] <= tolerance:
        near_support = True
    elif abs(current_price - latest_pivot['support3']) / latest_pivot['support3'] <= tolerance:
        near_support = True

    if abs(current_price - latest_pivot['resistance1']) / latest_pivot['resistance1'] <= tolerance:
        near_resistance = True
    elif abs(current_price - latest_pivot['resistance2']) / latest_pivot['resistance2'] <= tolerance:
        near_resistance = True
    elif abs(current_price - latest_pivot['resistance3']) / latest_pivot['resistance3'] <= tolerance:
        near_resistance = True

    return {
        'near_support': near_support,
        'near_resistance': near_resistance,
        'pivot_point': latest_pivot['pivot_point'],
        'support_levels': [latest_pivot['support1'], latest_pivot['support2'], latest_pivot['support3']],
        'resistance_levels': [latest_pivot['resistance1'], latest_pivot['resistance2'], latest_pivot['resistance3']]
    }


import pandas as pd
import numpy as np

def detect_market_trend(df, short_window=10, long_window=30):
    """
    Detects market trend using moving averages, ADX, and price action analysis.

    Args:
    - df: Pandas DataFrame containing market data (with columns 'close', 'high', 'low', 'open').
    - short_window: The window period for the short-term moving average.
    - long_window: The window period for the long-term moving average.

    Returns:
    - trend_signal: "uptrend", "downtrend", or "no_trend" indicating the market trend.
    """
    
    ### Step 1: Moving Average Crossover ###
    
    # Short-term (fast) and Long-term (slow) moving averages
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()

    # Moving average crossover strategy
    df['ma_signal'] = np.where(df['ma_short'] > df['ma_long'], 1, -1)
    return df['ma_signal'].iloc[-1]
    
    ### Step 2: ADX Calculation ###
    
def calculate_adx(df, n=14):
    """Calculates ADX (Average Directional Index) and returns ADX values."""
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
        
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
    tr1 = pd.DataFrame(df['high'] - df['low'])
    tr2 = pd.DataFrame(abs(df['high'] - df['close'].shift(1)))
    tr3 = pd.DataFrame(abs(df['low'] - df['close'].shift(1)))
    true_range = pd.concat([tr1, tr2, tr3], axis=1, ignore_index=True).max(axis=1)
    atr = true_range.rolling(window=n).mean()

    plus_di = pd.Series(plus_dm).rolling(window=n).mean() * 100 / atr
    minus_di = pd.Series(minus_dm).rolling(window=n).mean() * 100 / atr
        
    dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=n).mean()
        
    df['adx'] = adx
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    return df['adx'].iloc[-1], df['plus_di'].iloc[-1], df['minus_di'].iloc[-1]

    
    ### Step 3: Price Action Analysis ###
def price_action_analysis(df):
    """Analyzes price action using higher highs and higher lows for trend detection."""
    # Detect higher highs and higher lows for uptrend, and lower highs and lower lows for downtrend
    df['higher_highs'] = np.where((df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1)), 1, 0)
    df['lower_lows'] = np.where((df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1)), -1, 0)
    return df
    

def detect_trend(df):
    df = price_action_analysis(df)
    
    ### Step 4: Trend Detection Logic ###
    latest_ma_signal = detect_market_trend(df)
    latest_adx, latest_plus_di, latest_minus_di = calculate_adx(df)
    
    # ADX value threshold for trend strength
    adx_threshold = 25

    if latest_adx > adx_threshold:
        # Strong trend, check for +DI and -DI to confirm direction
        if latest_plus_di > latest_minus_di and latest_ma_signal == 1:
            return "uptrend"
        elif latest_minus_di > latest_plus_di and latest_ma_signal == -1:
            return "downtrend"
    else:
        # Weak or no trend based on ADX
        return "no_trend"

    ### Step 5: Confirming with Price Action ###
    if df['higher_highs'].iloc[-1] == 1:
        return "uptrend"
    elif df['lower_lows'].iloc[-1] == 1:
        return "downtrend"
    
    return "no_trend"



# def calculate_signal_strength(trend, rsi_value, pivot_point_data, ma_proximity, bb_signal, ma_support_resistance, bb_support_resistance, spike_indices, current_index):
#     """
#     Calculate the strength of the buy/sell signal based on trend, RSI, proximity to pivot points,
#     and support/resistance levels for MA and Bollinger Bands, including spike detection.

#     Args:
#     - trend: The current trend ('uptrend', 'downtrend', 'sideways')
#     - rsi_value: The RSI value to determine overbought/oversold conditions.
#     - pivot_point_data: Data on pivot points to check proximity to support/resistance.
#     - ma_proximity: A boolean indicating whether price is near the moving average.
#     - bb_signal: Bollinger Band signal ('upper_band', 'lower_band', or 'neutral')
#     - ma_support_resistance: Whether MA is acting as support or resistance ('support', 'resistance', 'neutral')
#     - bb_support_resistance: Whether Bollinger Bands are acting as support or resistance ('support', 'resistance', 'neutral')
#     - spike_indices: List of indices where spikes were detected.
#     - current_index: The index of the current price point for evaluating spike influence.

#     Returns:
#     - Signal strength as an integer between 0 and 100.
#     """
#     strength = 0

#     # Base on trend direction
#     if trend == 'uptrend':
#         strength += 30
#     elif trend == 'downtrend':
#         strength += 30

#     # Check RSI
#     if rsi_value < 30:  # Oversold, potential buy signal
#         strength += 20
#     elif rsi_value > 70:  # Overbought, potential sell signal
#         strength += 20

#     # Check if price is near a moving average
#     if ma_proximity:
#         strength += 10

#     # MA support/resistance
#     if ma_support_resistance == 'support':
#         strength += 10
#     elif ma_support_resistance == 'resistance':
#         strength += 10

#     # Check Bollinger Band signals
#     if bb_signal == 'upper_band':
#         strength += 10
#     elif bb_signal == 'lower_band':
#         strength += 10

#     # Bollinger Bands support/resistance
#     if bb_support_resistance == 'support':
#         strength += 10
#     elif bb_support_resistance == 'resistance':
#         strength += 10

#     # Pivot point proximity
#     if pivot_point_data['support_1'] <= strength <= pivot_point_data['resistance_1']:
#         strength += 20

#     # Check for spikes
#     if current_index in spike_indices:
#         strength += 20  # Increase strength if a spike is detected at the current index

#     return min(strength, 100)  # Cap the strength at 100


def calculate_signal_strength(
    df,
    trend, 
    rsi_value, 
    pivot_point_data, 
    bb_signal, 
    ma_support_resistance,
    ma48_support_resistance, 
    bb_support_resistance, 
    spike_indices, 
    current_index
):
    """
    Calculates the strength of a buy/sell signal based on various indicators.

    Args:
    - trend: The current market trend (bullish or bearish).
    - rsi_value: The current RSI value (e.g., 30 = oversold, 70 = overbought).
    - pivot_point_data: Data regarding the nearest pivot points (S1, R1, etc.).
    - ma_proximity: Whether the price is near the moving average.
    - bb_signal: Whether the price is near the upper or lower Bollinger Band.
    - ma_support_resistance: Whether the moving average is acting as support or resistance.
    - bb_support_resistance: Whether the Bollinger Band is acting as support or resistance.
    - spike_indices: Indices where spikes are detected.
    - current_index: The index of the current bar being analyzed.

    Returns:
    - A signal strength value or category indicating strong buy, weak buy, weak sell, or strong sell.
    """

    # Initialize signal strength to neutral
    strength = 0.5  # Start with neutral strength (0.5 represents neutral, 0 strong sell, 1 strong buy)

    ### Moving Average (MA) Proximity and Support/Resistance ###
    if ma_support_resistance == 'support':
        strength += 0.1  # Stronger buy signal if the moving average acts as support
    elif ma_support_resistance == 'resistance':
        strength -= 0.1  # Stronger sell signal if the moving average acts as resistance


    if ma48_support_resistance == 'support':
        strength += 0.1  # Stronger buy signal if the moving average acts as support
    elif ma48_support_resistance == 'resistance':
        strength -= 0.1  # Stronger sell signal if the moving average acts as resistance
    
    # # If price is near the moving average, slightly boost the signal strength
    # if ma_proximity:
    #     strength += 0.05

    # if ma48_proximity:
    #     strength += 0.05

    ### Bollinger Band (BB) Proximity and Support/Resistance ###
    if bb_support_resistance == 'support':
        strength += 0.15  # Strong buy if the Bollinger Band acts as support
    elif bb_support_resistance == 'resistance':
        strength -= 0.15  # Strong sell if the Bollinger Band acts as resistance
    
    # Boost the signal based on proximity to upper/lower Bollinger Bands
    if bb_signal == 'upper_band':
        strength -= 0.1  # Selling pressure at upper Bollinger Band
    elif bb_signal == 'lower_band':
        strength += 0.1  # Buying pressure at lower Bollinger Band

    ### Spike Detection ###
    if len(spike_indices) > 0 and spike_indices[-1] == current_index:
        # If the last spike occurred at the current price bar
        strength -= 0.2  # Spikes in Boom/Crash markets usually suggest a reversal

    ### RSI Indicator ###
    if rsi_value < 20:
        strength += 0.1  # Oversold, so stronger buy signal
    elif rsi_value > 80:
        strength -= 0.1  # Overbought, so stronger sell signal



    pivot_point_data = get_pivot_point_data(df, current_price=df['close'].iloc[-1])

    if pivot_point_data['near_support']:
        strength += 0.15  # Stronger buy signal if price is near support levels
    elif pivot_point_data['near_resistance']:
        strength -= 0.15  # Stronger sell signal if price is near resistance levels

    ### Trend Detection ###
    if trend == 'uptrend':
        strength += 0.1  # Bullish trend enhances buy signal
    elif trend == 'downtrend':
        strength -= 0.1  # Bearish trend enhances sell signal



    # Normalize the signal strength to be between 0 and 1
    strength = max(0, min(1, strength))

    return strength





class MyStrategy():
    def __init__(self, data):
        self.data = data
        bt = Indicator(self.data)
        # Initialization of indicators and price data
        self.ma = bt.moving_average(period=10)
        self.rsi = bt.rsi(period=14)
        self.df = self.data  # Placeholder for DataFrame with price data
        
    async def run(self):
        # 1. Calculate pivot points
        pivot_point_data = calculate_pivot_points(self.data)

        spike_indices = detect_spike(self.df, window=20, threshold=1.5)
        current_index = len(self.df) - 1
        
        # 2. Detect the current trend
        trend = detect_trend(self.df)
        
        # 3. Check RSI
        
        rsi_value = self.rsi.iloc[-1]
        
        # 4. Check if price is near MA
        # ma_proximity = await is_price_near_ma(self.df, ma_period=10, tolerance=0.01)

        # ma48_proximity = await is_price_near_ma(self.df, ma_period=48, tolerance=0.01)
        
        # 5. Check Bollinger Band signal
        bb_signal = await is_price_near_bollinger_band(self.df, period=20, std_dev=2, tolerance=0.01)

        bb_support_resistance = await is_bollinger_band_support_resistance(self.df)
        ma_support_resistance  = await is_support_resistance(self.df, 10)
        ma48_support_resistance  = await is_support_resistance(self.df, 48)
        
        # 6. Calculate the signal strength
        signal_strength = calculate_signal_strength(
            self.df,
            trend,
            rsi_value,
            pivot_point_data,
            bb_signal,
            ma_support_resistance,
            ma48_support_resistance,
            bb_support_resistance,
            spike_indices=spike_indices,
            current_index=current_index
        )
            
        
        # # 7. Execute trade based on signal strength
        # if signal_strength >= 0.7:
        #     # Strong buy signal
        #     return 
        # elif signal_strength <= 0.2:
        #     # Strong sell signal
        #     return "SELL"
        return signal_strength
    

async def combine_timeframe_signals(timeframes, weights=None):
        """
        Combines the signal strengths from multiple timeframes using weighted averaging.

        Args:
        - m5_strength: Signal strength from the M5 timeframe (0 to 1).
        - m15_strength: Signal strength from the M15 timeframe (0 to 1).
        - m30_strength: Signal strength from the M30 timeframe (0 to 1).
        - m5_weight: Weight for M5 signal strength.
        - m15_weight: Weight for M15 signal strength.
        - m30_weight: Weight for M30 signal strength.

        Returns:
        - Combined signal strength and its category.
        """

        # Weighted average of signal strengths
        # combined_strength = (m5_strength * m5_weight +
        #                     m15_strength * m15_weight +
        #                     m30_strength * m30_weight)
        
        if weights is None:
            weights = Config.WEIGHTS.values()

        combined_strength = 0
        for tf, weight in zip(timeframes, weights):
            combined_strength += tf * weight
        # Determine the final signal category
        if combined_strength >= 0.8:
            final_signal = 'strong buy'
        elif 0.6 <= combined_strength < 0.8:
            final_signal = 'weak buy'
        elif 0.4 <= combined_strength < 0.6:
            final_signal = 'neutral'
        elif 0.2 <= combined_strength < 0.4:
            final_signal = 'weak sell'
        else:
            final_signal = 'strong sell'

        return combined_strength, final_signal


