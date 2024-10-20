# from utils.indicators import Indicator
# from ResistanceSupportDectector.detector import (
#     ma_support_resistance, check_ema, is_support_resistance, 
#     is_bollinger_band_support_resistance, is_price_near_bollinger_band, 
#     is_price_near_ma
# )
# import asyncio
# from ResistanceSupportDectector.spikeDectector import detect_spikes
# from ResistanceSupportDectector.aiStartegy import MyStrategy, combine_timeframe_signals

# class Strategy:

#     @classmethod
#     async def runStrategy(cls, df, tolerance, breakout_threshold=0.0035):
#         """
#         Generates a buy or sell signal based on various indicators like MA10, MA48, EMA, and Bollinger Bands.
#         Prioritizes signals in case of conflicts and checks for breakouts.

#         Args:
#             df: Pandas DataFrame containing price data with columns 'Close' and 'Date'.
#             tolerance: Percentage tolerance for considering price near key indicators.
#             breakout_threshold: Percentage threshold for price breakout beyond an indicator.

#         Returns:
#             "BUY", "SELL", or "HOLD" based on the signal evaluation.
#         """
#         indicator = Indicator(df)

#         # Fetch indicators
#         # ma10 = indicator.moving_average(10)
#         # ma48 = indicator.moving_average(48)

#         # Get conditions for moving averages, EMA, and Bollinger Bands
#         # ma10_behavior = await is_support_resistance(df, 10)
#         # ma48_behavior = await is_support_resistance(df, 48)
#         ema_behavior = await check_ema(df, period=200, tolerance=tolerance, breakout_value=breakout_threshold)
#         bb_behavior = await is_bollinger_band_support_resistance(df)
#         price_near_bb = await is_price_near_bollinger_band(df, tolerance=tolerance)

#         # Additional checks for proximity to MAs with breakout consideration
#         price_near_ma48 = await is_price_near_ma(df, tolerance, breakout_threshold, ma_period=48)
#         price_near_ma10 = await is_price_near_ma(df, tolerance, breakout_threshold, ma_period=10)
#         #breakout_48 = df['close'].iloc[-1] > ma48.iloc[-1] * (1 + breakout_threshold)

#         # Define buy and sell conditions
#         buy_conditions = [
#             price_near_ma10 == 'support',
#             price_near_ma48 == 'support',
#             ema_behavior == 'support',
#             bb_behavior == 'support' and price_near_bb == 'lower_band'
#         ]
#         sell_conditions = [
#             price_near_ma10 == 'resistance',
#             price_near_ma48 == 'resistance',
#             ema_behavior == 'resistance',
#             bb_behavior == 'resistance' and price_near_bb == 'upper_band'
#         ]

#         # Priority-based logic to handle mixed signals more effectively
#         print(f"Buy Conditions: {buy_conditions}, Sell Conditions: {sell_conditions}")
#         buy_count = buy_conditions.count(True)
#         sell_count = sell_conditions.count(True)

#         # Decision making based on buy and sell condition count
#         if buy_count > sell_count:
#             #print("Buy signal stronger based on conditions.")
#             return "BUY"
#         elif sell_count > buy_count:
#             #print("Sell signal stronger based on conditions.")
#             return "SELL"
#         else:
#             #print("No clear signal, holding position.")
#             return "HOLD"

#     @classmethod
#     async def process_multiple_timeframes(cls, dataframes, ma_period=10, tolerance=0.0035, breakout_threshold=0.0015, std_dev=2):
#         """
#         Processes multiple timeframes to generate a combined buy or sell signal. 
#         Adjusts tolerance dynamically based on volatility from different timeframes.

#         Args:
#             dataframes: A list of pandas DataFrames for different timeframes (e.g., M15, M30, H1).
#             ma_period: Length of the short-term moving average.
#             tolerance: Percentage tolerance for price proximity to indicators.
#             breakout_threshold: Threshold for breakout detection.
#             std_dev: Standard deviation for Bollinger Bands.

#         Returns:
#             A tuple containing the final signal ("BUY", "SELL", or "HOLD"), the overall signal strength, and detailed signals from each timeframe.
#         """
#         overall_volatility = calculate_overall_volatility_from_df(dataframes)
#         tolerance_adjusted = tolerance * (1 + overall_volatility)

#         #print(f"Adjusted tolerance based on overall volatility: {tolerance_adjusted}, {overall_volatility}")

#         # tasks = []
#         # for df in dataframes:
#         #     strategy_instance = MyStrategy(df)
#         #     tasks.append(asyncio.create_task(cls.runStrategy(df, tolerance_adjusted, breakout_threshold)))
        
#         # result_signals = await asyncio.gather(*tasks)
#         # combined_strength, combined_signal = await combine_timeframe_signals(result_signals)

#         #print(f"Timeframe Results: {result_signals}, Combined Strength: {combined_strength}")

#         tasks = []
#         task2 = []
#         for df in dataframes:
#             startegy = MyStrategy(df)
#             task2.append(asyncio.create_task(cls.runStrategy(df, float(tolerance_adjusted), breakout_threshold)))
#             tasks.append(asyncio.create_task(startegy.run()))

#         result_signals = await asyncio.gather(*task2) # type: ignore
#         results = await asyncio.gather(*tasks)

#         strength, signal = await combine_timeframe_signals(results)

#         if all(signal == "BUY" for signal in result_signals):
#             return [1, strength, result_signals]
#         elif all(signal == "SELL" for signal in result_signals):
#             return [0, strength, result_signals]
#         elif result_signals.count("BUY") > result_signals.count("SELL"):
#             return [1, strength, result_signals]
#         elif result_signals.count("SELL") > result_signals.count("BUY"):
#             return [0, strength, result_signals]
#         else:
#             return [1, strength, result_signals]  # Hold when signals are mixed

# # Utility functions for volatility calculations
# import pandas as pd
# import numpy as np

# def calculate_volatility_from_df(df):
#     """
#     Calculate volatility (standard deviation of percentage price changes).
    
#     Args:
#         df: DataFrame containing price data.
    
#     Returns:
#         Volatility as a percentage.
#     """
#     returns = df['close'].pct_change().dropna()
#     volatility = np.std(returns) * 100
#     return volatility

# def calculate_overall_volatility_from_df(dataframes, weights=(0.4, 0.3, 0.3)):
#     """
#     Calculate weighted overall volatility based on multiple timeframes.
    
#     Args:
#         dataframes: List of DataFrames, each representing a different timeframe.
#         weights: Weights to assign to each timeframe.
    
#     Returns:
#         The weighted overall market volatility.
#     """

#     m15_volatility = calculate_volatility_from_df(dataframes[0])
#     m30_volatility = calculate_volatility_from_df(dataframes[1])
#     h1_volatility = calculate_volatility_from_df(dataframes[2])

#     overall_volatility = (weights[0] * m15_volatility) + (weights[1] * m30_volatility) + (weights[2] * h1_volatility)
#     return overall_volatility


#====================================================================================================================================================




from utils.indicators import Indicator
from ResistanceSupportDectector.detector import (
    check_ema, is_bollinger_band_support_resistance, 
    is_price_near_bollinger_band, is_price_near_ma
)
import asyncio
from ResistanceSupportDectector.spikeDectector import detect_spikes
from ResistanceSupportDectector.aiStartegy import MyStrategy, combine_timeframe_signals
# import pandas_ta as ta
class Strategy:

    @classmethod
    async def runStrategy(cls, df, tolerance, breakout_threshold=0.0035):
        """
        Generates a buy or sell signal based on various indicators like MA10, MA48, EMA, and Bollinger Bands.
        Prioritizes spike detection and filters signals accordingly.

        Args:
            df: Pandas DataFrame containing price data with columns 'Close', 'High', 'Low', and 'Date'.
            tolerance: Percentage tolerance for considering price near key indicators.
            breakout_threshold: Percentage threshold for price breakout beyond an indicator.

        Returns:
            "BUY", "SELL", or "HOLD" based on the signal evaluation and spike detection.
        """
        indicator = Indicator(df)

        # Fetch indicators
        ema_behavior = await check_ema(df, period=200, tolerance=tolerance, breakout_value=breakout_threshold)
        bb_behavior = await is_bollinger_band_support_resistance(df)
        price_near_bb = await is_price_near_bollinger_band(df, tolerance=tolerance)

        # Additional checks for proximity to MAs with breakout consideration
        price_near_ma48 = await is_price_near_ma(df, tolerance, breakout_threshold, ma_period=48)
        price_near_ma10 = await is_price_near_ma(df, tolerance, breakout_threshold, ma_period=10)

        # Detect spikes
        spike_detected = await detect_spikes(df)

        # Define buy and sell conditions with additional spike handling
        buy_conditions = [
            price_near_ma10 == 'support',
            price_near_ma48 == 'support',
            ema_behavior == 'support',
            bb_behavior == 'support' and price_near_bb == 'lower_band',
            spike_detected == "spike_down"  # Spike down detected, consider buying
        ]
        sell_conditions = [
            price_near_ma10 == 'resistance',
            price_near_ma48 == 'resistance',
            ema_behavior == 'resistance',
            bb_behavior == 'resistance' and price_near_bb == 'upper_band',
            spike_detected == "spike_up"  # Spike up detected, consider selling
        ]

        # Apply noise filtering based on volatility
        atr_value = calculate_atr(df)
        noise_filter = atr_value > tolerance  # Only act if market volatility is significant

        # Decision-making logic
        buy_score = buy_conditions.count(True)
        sell_score = sell_conditions.count(True)
        confidence_score = buy_score - sell_score  # Positive for buy, negative for sell

        # Only proceed with high volatility (noise filtering)
        if noise_filter:
            if confidence_score > 0:
                return "BUY"
            elif confidence_score < 0:
                return "SELL"
        return "HOLD"

    @classmethod
    async def process_multiple_timeframes(cls, dataframes, symbol, ma_period=10, tolerance=0.003, breakout_threshold=0.0035, std_dev=2):
        """
        Processes multiple timeframes to generate a combined buy or sell signal. 
        Adjusts tolerance dynamically based on volatility and catches spikes across different timeframes.

        Args:
            dataframes: A list of pandas DataFrames for different timeframes (e.g., M15, M30, H1).
            ma_period: Length of the short-term moving average.
            tolerance: Percentage tolerance for price proximity to indicators.
            breakout_threshold: Threshold for breakout detection.
            std_dev: Standard deviation for Bollinger Bands.

        Returns:
            A tuple containing the final signal ("BUY", "SELL", or "HOLD"), the overall signal strength, and detailed signals from each timeframe.
        """
        overall_volatility = calculate_overall_volatility_from_df(dataframes)
        tolerance_adjusted = tolerance * (1 + overall_volatility)
        features = calculate_features(dataframes[0], dataframes[1], dataframes[2])
        # path = 'C:/Users/Admin/codynego/mlforfinance/forexBot/Boom500_rf.pkl'
        
        model_paths = {
            "BOOM1000": 'models/Boom1000_rf.pkl',
            "BOOM500": 'models/Boom500_rf.pkl',
            "CRASH500": 'models/Crash500_rf.pkl',
            "CRASH1000": 'models/Crash1000_rf.pkl',
        }


        if symbol not in model_paths:
            print("Invalid symbol")
            return None

        path = model_paths[symbol]

        # load model using joblib
        model = joblib.load(path)
        ai_tolerance = model.predict(features)[0]
        #print(ai_tolerance)

        tasks = []
        task2 = []
        for df, tol in zip(dataframes, ai_tolerance):
            strategy = MyStrategy(df)
            task2.append(asyncio.create_task(cls.runStrategy(df, float(tol), breakout_threshold)))
            tasks.append(asyncio.create_task(strategy.run()))
            #print("new timeframes: ==============================================")

        result_signals = await asyncio.gather(*task2)  # Signals from runStrategy
        results = await asyncio.gather(*tasks)  # Signals from MyStrategy.run()

        strength, signal = await combine_timeframe_signals(results)

        # Determine final decision based on combined signals
        if all(signal == "BUY" for signal in result_signals):
            return [1, strength, result_signals]
        elif all(signal == "SELL" for signal in result_signals):
            return [0, strength, result_signals]
        elif result_signals.count("BUY") > result_signals.count("SELL"):
            return [1, strength, result_signals]
        elif result_signals.count("SELL") > result_signals.count("BUY"):
            return [0, strength, result_signals]
        else:
            return [1, strength, result_signals]  # Hold when signals are mixed

# Utility function for ATR and volatility calculations
import pandas as pd
import numpy as np

def calculate_atr(df, period=14):
    """
    Calculate the Average True Range (ATR) to measure volatility.

    Args:
        df: DataFrame containing price data.
        period: Period over which to calculate the ATR.

    Returns:
        ATR value.
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean().iloc[-1]
    return atr

def calculate_volatility_from_df(df):
    """
    Calculate volatility (standard deviation of percentage price changes).
    
    Args:
        df: DataFrame containing price data.
    
    Returns:
        Volatility as a percentage.
    """
    returns = df['close'].pct_change().dropna()
    volatility = np.std(returns) * 100
    return volatility
import joblib

def calculate_overall_volatility_from_df(dataframes, weights=(0.4, 0.3, 0.3)):
    """
    Calculate weighted overall volatility based on multiple timeframes.
    
    Args:
        dataframes: List of DataFrames, each representing a different timeframe.
        weights: Weights to assign to each timeframe.
    
    Returns:
        The weighted overall market volatility.
    """
    m15_volatility = calculate_volatility_from_df(dataframes[0])
    m30_volatility = calculate_volatility_from_df(dataframes[1])
    h1_volatility = calculate_volatility_from_df(dataframes[2])

    overall_volatility = (weights[0] * m15_volatility) + (weights[1] * m30_volatility) + (weights[2] * h1_volatility)
    return overall_volatility



import pandas as pd

# def calculate_features(m5_data, m15_data, h1_data):

#     # Calculate indicators for M5 timeframe
#     m5_data['close_m5'] = m5_data['close']
#     m5_data['MA10_m5'] = ta.sma(m5_data['close'], length=10)
#     m5_data['MA48_m5'] = ta.sma(m5_data['close'], length=48)
#     m5_data['EMA200_m5'] = ta.ema(m5_data['close'], length=200)
#     m5_data[['BB_Low_m5', 'BB_Mid_m5', 'BB_High_m5' ]] = ta.bbands(m5_data['close'], length=20, std=2).iloc[:, :3]


#     # Calculate indicators for M15 timeframe
#     m15_data['close_m15'] = m15_data['close']
#     m15_data['MA10_m15'] = ta.sma(m15_data['close'], length=10)
#     m15_data['MA48_m15'] = ta.sma(m15_data['close'], length=48)
#     m15_data['EMA200_m15'] = ta.ema(m15_data['close'], length=200)
#     m15_data[['BB_Low_m15','BB_Mid_m15', 'BB_High_m15']] = ta.bbands(m15_data['close'], length=20, std=2).iloc[:, :3]

#     # Calculate indicators for H1 timeframe
#     h1_data['close_h1'] = h1_data['close']
#     h1_data['MA10_h1'] = ta.sma(h1_data['close'], length=10)
#     h1_data['MA48_h1'] = ta.sma(h1_data['close'], length=48)
#     h1_data['EMA200_h1'] = ta.ema(h1_data['close'], length=200)
#     h1_data[['BB_High_h1', 'BB_Mid_h1', 'BB_Low_h1']]= ta.bbands(h1_data['close'], length=20).iloc[:, :3]

#     # Calculate timeframe tolerances (this can be adjusted depending on how you calculate tolerance)
#     m5_data['Timeframe_Tolerance_M5'] = abs(m5_data['close'] - m5_data['MA10_m5']) / m5_data['MA10_m5'] * 100
#     m15_data['Timeframe_Tolerance_M15'] = abs(m15_data['close'] - m15_data['MA10_m15']) / m15_data['MA10_m15'] * 100
#     h1_data['Timeframe_Tolerance_h1'] = abs(h1_data['close'] - h1_data['MA10_h1']) / h1_data['MA10_h1'] * 100

#     # Merge dataframes on index
#     combined_data = pd.concat([m5_data, m15_data, h1_data], axis=1, join='inner')

#     # Select relevant features
#     features = combined_data[['close_m5', 'close_m15', 'close_h1', 'MA10_m5',
#                              'MA48_m5', 'EMA200_m5', 'BB_High_m5', 'BB_Low_m5',
#                              'MA10_m15', 'MA48_m15', 'EMA200_m15', 'BB_High_m15', 'BB_Low_m15',
#                              'MA10_h1', 'MA48_h1', 'EMA200_h1', 'BB_High_h1', 'BB_Low_h1',
#                              'Timeframe_Tolerance_M5', 'Timeframe_Tolerance_M15', 'Timeframe_Tolerance_h1']]

#     return features.tail(1)


import pandas as pd

def calculate_features(m5_data, m15_data, h1_data):
    # Calculate indicators for M5 timeframe
    m5_data['close_m5'] = m5_data['close']
    m5_data['MA10_m5'] = m5_data['close'].rolling(window=10).mean()
    m5_data['MA48_m5'] = m5_data['close'].rolling(window=48).mean()
    m5_data['EMA200_m5'] = m5_data['close'].ewm(span=200, adjust=False).mean()
    m5_data['BB_Mid_m5'] = m5_data['close'].rolling(window=20).mean()
    m5_data['BB_STD_m5'] = m5_data['close'].rolling(window=20).std()
    m5_data['BB_Low_m5'] = m5_data['BB_Mid_m5'] - (m5_data['BB_STD_m5'] * 2)
    m5_data['BB_High_m5'] = m5_data['BB_Mid_m5'] + (m5_data['BB_STD_m5'] * 2)

    # Calculate indicators for M15 timeframe
    m15_data['close_m15'] = m15_data['close']
    m15_data['MA10_m15'] = m15_data['close'].rolling(window=10).mean()
    m15_data['MA48_m15'] = m15_data['close'].rolling(window=48).mean()
    m15_data['EMA200_m15'] = m15_data['close'].ewm(span=200, adjust=False).mean()
    m15_data['BB_Mid_m15'] = m15_data['close'].rolling(window=20).mean()
    m15_data['BB_STD_m15'] = m15_data['close'].rolling(window=20).std()
    m15_data['BB_Low_m15'] = m15_data['BB_Mid_m15'] - (m15_data['BB_STD_m15'] * 2)
    m15_data['BB_High_m15'] = m15_data['BB_Mid_m15'] + (m15_data['BB_STD_m15'] * 2)

    # Calculate indicators for H1 timeframe
    h1_data['close_h1'] = h1_data['close']
    h1_data['MA10_h1'] = h1_data['close'].rolling(window=10).mean()
    h1_data['MA48_h1'] = h1_data['close'].rolling(window=48).mean()
    h1_data['EMA200_h1'] = h1_data['close'].ewm(span=200, adjust=False).mean()
    h1_data['BB_Mid_h1'] = h1_data['close'].rolling(window=20).mean()
    h1_data['BB_STD_h1'] = h1_data['close'].rolling(window=20).std()
    h1_data['BB_Low_h1'] = h1_data['BB_Mid_h1'] - (h1_data['BB_STD_h1'] * 2)
    h1_data['BB_High_h1'] = h1_data['BB_Mid_h1'] + (h1_data['BB_STD_h1'] * 2)

    # Calculate timeframe tolerances
    m5_data['Timeframe_Tolerance_M5'] = abs(m5_data['close'] - m5_data['MA10_m5']) / m5_data['MA10_m5'] * 100
    m15_data['Timeframe_Tolerance_M15'] = abs(m15_data['close'] - m15_data['MA10_m15']) / m15_data['MA10_m15'] * 100
    h1_data['Timeframe_Tolerance_h1'] = abs(h1_data['close'] - h1_data['MA10_h1']) / h1_data['MA10_h1'] * 100

    # Merge dataframes on index
    combined_data = pd.concat([m5_data, m15_data, h1_data], axis=1, join='inner')

    # Select relevant features
    features = combined_data[['close_m5', 'close_m15', 'close_h1', 
                               'MA10_m5', 'MA48_m5', 'EMA200_m5', 
                               'BB_High_m5', 'BB_Low_m5',
                               'MA10_m15', 'MA48_m15', 'EMA200_m15', 
                               'BB_High_m15', 'BB_Low_m15',
                               'MA10_h1', 'MA48_h1', 'EMA200_h1', 
                               'BB_High_h1', 'BB_Low_h1',
                               'Timeframe_Tolerance_M5', 
                               'Timeframe_Tolerance_M15', 
                               'Timeframe_Tolerance_h1']]

    return features.tail(1)
