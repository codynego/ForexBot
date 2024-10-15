from utils.indicators import Indicator
#from ResistanceSupportDectector.detector import generate_buy_signal
from ResistanceSupportDectector.detector import ma_support_resistance, check_ema, is_support_resistance, is_bollinger_band_support_resistance, is_price_near_bollinger_band, is_price_near_ma
import asyncio
from ResistanceSupportDectector.spikeDectector import detect_spikes
from ResistanceSupportDectector.aiStartegy import MyStrategy, combine_timeframe_signals

class Strategy:
    @classmethod
    # def rsiStrategy(cls, data):
    #     if generate_buy_signal(data):
    #         return -1
    #     else:
    #         return 1
    #     # indicator = Indicator(data)
    #     # rsi = indicator.rsi(14)
    #     # ma = indicator.moving_average(10)
    #     # print('moving average',ma.tail(1))
    #     # rsi_value = rsi.tail(1).values[0]
    #     # #print(data['close'].tail(1))
    #     # if rsi_value > 30:
    #     #     return 1
    #     # elif rsi_value < 30:
    #     #     return -1
    #     # return 0


    @classmethod
    async def runStrategy(cls, df, tolerance=0.005, breakout_threshold=0.025):
        """
        Generates a buy signal based on MA10 behavior and price proximity.

        Args:
            df: Pandas DataFrame containing price data with columns 'Close' and 'Date'.
            ma_period: Length of the moving average.
            tolerance: Percentage tolerance for considering price near MA.
            breakout_threshold: Percentage threshold for price breakout.

        Returns:
            True if a buy signal is generated, False otherwise.
        """
        #spikes = detect_spikes(df)
        # print(len(spikes))
        indicator = Indicator(df)
        ma48 = indicator.moving_average(48)
        # check m0ving average 10 behavior
        # ma10_behavior = await is_support_resistance(df, ma_period)
        #price_near_ma10 = await is_price_near_ma(df, ma_period, tolerance)
        # breakout_10 = df['close'].iloc[-1] > ma48.iloc[-1] * (1 + breakout_threshold)

        ma10_behavior = await is_support_resistance(df, 10)
        ma48_behavior = await is_support_resistance(df, 48)

        # check moving average 48 behavior

        ma48_period = 48
        #ma48_behavior = await is_support_resistance(df, 48)
        price_near_ma48 = await is_price_near_ma(df,tolerance, ma_period=48)
        price_near_ma10 = await is_price_near_ma(df,tolerance, ma_period=10)
        breakout_48 = df['close'].iloc[-1] > ma48.iloc[-1] * (1 + breakout_threshold)

        # check bolling band behavior
        #ma48_period = 48
        bb_behavior = await is_bollinger_band_support_resistance(df)
        price_near_bb = await is_price_near_bollinger_band(df, tolerance=tolerance)
        #breakout_48 = df['close'].iloc[-1] > ma48.iloc[-1] * (1 + breakout_threshold)

        ema_behaviour = await check_ema(df, period=200, tolerance=tolerance)


        buy_conditions = [
            price_near_ma10 == 'support',
            price_near_ma48 == 'support',
            ema_behaviour == 'support',
            bb_behavior == 'support' and price_near_bb == 'lower_band'
        
        ]
        # print("buy  conditions: ", buy_conditions)

    
        #print("buy_condition", buy_conditions)
        sell_conditions = [
            price_near_ma10 == 'resistance',
            price_near_ma48 == 'resistance',
            ema_behaviour == 'resistance',
            bb_behavior == 'resistance' and price_near_bb == 'upper_band'
        ]
        # print("sell _condition", sell_conditions)
        # print("=========================================")
        
        if any(buy_conditions) and not any(sell_conditions):

            return "BUY"
        
        elif any(sell_conditions):
            return "SELL"
        else:
            return "HOLD"
    

        

    @classmethod
    async def process_multiple_timeframes(cls, dataframes, ma_period=10, tolerance=0.005, breakout_threshold=0.015, std_dev=2):
        """
        Processes multiple timeframes to generate a buy or sell signal.

        Args:
            dataframes: A list of pandas DataFrames, one for each timeframe.
            ma_period: Length of the short-term moving average.
            tolerance: Percentage tolerance for considering price near MA.
            breakout_threshold: Percentage threshold for price breakout.
            std_dev: Number of standard deviations for Bollinger Bands.

        Returns:
            "BUY", "SELL", or "HOLD" based on the combined signals from all timeframes.
        """
        
        tasks = []
        task2 = []
        for df in dataframes:
            startegy = MyStrategy(df)
            task2.append(asyncio.create_task(cls.runStrategy(df,tolerance, breakout_threshold)))
            tasks.append(asyncio.create_task(startegy.run()))

        result2 = await asyncio.gather(*task2) # type: ignore
        results = await asyncio.gather(*tasks)
        strength, signal = await combine_timeframe_signals(results)
                #Check if all signals are the same
        # if all(result == "BUY" for result in results):
        #     return 1
        # elif all(result == "SELL" for result in results):
        #     return -1
        # else:
        #     return 0
        # print(result2)
 
        if all(result[0] == "BUY" for result in result2):
            return [1, strength, result2]
        # elif strength >= 0.8:
        #     return [1, strength]
        elif all(result[0] == "SELL" for result in result2):
            return [-1, strength, result2]
        # elif strength <= 0.25:
        #     return [-1, strength]
        else:
            return [0, strength, result2]
        

        # Check if all signals are the same
        # if all(result == "BUY" for result in results):
        #     return 1
        # elif all(result == "SELL" for result in results):
        #     return -1
        # else:
        #     return 0