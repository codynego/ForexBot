import pandas as pd
from utils.indicators import Indicator
from utils.strategies import Strategy
import asyncio
from config import Config
import os
import django
from asgiref.sync import sync_to_async
from deriv_api import DerivAPI

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()
# Django setup

from traderbot.models import Market, Indicator as IndicatorModel, Signal

class TradingBot:
    def __init__(self, login, password, server):
        # self.login = login
        # self.password = password
        # self.server = server
        self.connected = False
        self.signals_cache = {}
        self.prev_predictions = {}
        self.pending_signals = {}
        self.opened_positions = {}
    
    # def connect(self):
    #     if not mt5.initialize(): # type: ignore
    #         print("initialize() failed, error code =", mt5.last_error()) # type: ignore
    #         mt5.shutdown() # type: ignore
    #         self.connected = False
    #         return self.connected
    #     authorized = mt5.login(self.login, password=self.password, server=self.server) # type: ignore
    #     self.connected = authorized
    #     return self.connected

    # def disconnect(self):
    #     mt5.shutdown() # type: ignore
    #     self.connected = False

    # async def fetch_data(self, symbol, timeframe, start, end):
    #     if not self.connected:
    #         raise Exception("Not connected to MT5")
    #     rates = mt5.copy_rates_range(symbol, timeframe, start, end) # type: ignore
    #     df = pd.DataFrame(rates)
    #     return df

    async def connect_deriv(self, app_id):
        api = DerivAPI(app_id=app_id)
        api_token = Config.DERIV_API_TOKEN

        authorize = await api.authorize(api_token)
        return authorize, api


    async def fetch_data_Deriv(self, api, symbol, timeframe):
        try:
            ticks = await api.ticks_history({
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 1000,
                "end": "latest",
                "style": "candles",
                "granularity": timeframe
                }
            )
            df = pd.DataFrame(list(ticks['candles']))
            df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
            return df
        except Exception as e:
            print("something went wrong", e)
            return None

    
    async def fetch_all_timeframes(self, api, market):

        data_tasks = [self.fetch_data_Deriv(api, market, timeframe) for timeframe in Config.TIME_FRAMES]
        return await asyncio.gather(*data_tasks)
    


    async def fetch_data_for_multiple_markets(self,api, markets):
        """Fetches data for multiple markets and timeframes concurrently.

        Args:
            markets: A list of market symbols.
            start: Start date for data retrieval.
            end: End date for data retrieval.
            timeframes: A list of timeframes.

        Returns:
            A dictionary of dataframes, where keys are market symbols and values are lists of dataframes (one for each timeframe).
        """

        data_tasks = [asyncio.create_task(self.fetch_all_timeframes(api, market)) for market in markets]
        return await asyncio.gather(*data_tasks)
            
    def apply_strategy(self, data, strategy):
        indicator = Indicator(data.head(14))
        calc = indicator.rsi()
        last_indicator_value = calc.tail(1).values[0]
        print(last_indicator_value)

    async def generate_signal(self, data, strategy="rsistrategy", symbol=None):
        price = data[0]['close'].iloc[-1] # type: ignore
        signal = {"symbol": symbol, "price": price, "type": None, "strength": None}

        if strategy == "rsistrategy":
            # stra = Strategy.rsiStrategy(data)
            stra, strength = await Strategy.process_multiple_timeframes(data)
     
            signal["strength"] = round(strength, 2)
            if stra == 1:
                signal["type"] = "BUY"
            elif stra == -1:
                signal["type"] = "SELL"
            elif stra == 0:
                signal["type"] = "HOLD"

            #Check for duplicate signals
            signal_key = (symbol, signal["type"])
            if signal_key in self.signals_cache:
                return None  # Duplicate found

            # Save the signal to the database
            # saved_signal = await self.save_to_database("Signal", symbol, signal)
                
            # Update cache
            self.signals_cache[signal_key] = signal
            return signal
            

    async def process_multiple_signals(self, data_list, market_list):
            # run signals concurretly
            signals = await asyncio.gather(*(self.generate_signal(data, symbol=market) for data, market in zip(data_list, market_list)))
            return signals

    # async def save_to_database(self, model, symbol, data):
    #     if model == "Market":
    #         market, created = await sync_to_async(Market.objects.get_or_create)(
    #             symbol=symbol, 
    #             defaults={
    #                 'open': data["open"], 
    #                 'high': data["high"], 
    #                 'low': data["low"], 
    #                 'close': data["close"], 
    #                 'volume': data["volume"]
    #             }
    #         )
    #         if created:
    #             await sync_to_async(market.save)()
    #         return market

    #     elif model == "Indicator":
    #         indicator, created = await sync_to_async(IndicatorModel.objects.get_or_create)(
    #             market=symbol, 
    #             defaults={
    #                 'rsi': data["rsi"], 
    #                 'macd': data["macd"], 
    #                 'bollinger_bands': data["bollinger_bands"], 
    #                 'moving_average': data["moving_average"]
    #             }
    #         )
    #         if created:
    #             await sync_to_async(indicator.save)()
    #         return indicator

    #     elif model == "Signal":
    #         signal, created = await sync_to_async(Signal.objects.get_or_create)(
    #             symbol=symbol, 
    #             price = data["price"],
    #             type = data["type"], 
    #             strength = data["strength"],
    #         )
    #         if created:
    #             await sync_to_async(signal.save)()
    #         return signal

    def signal_toString(self, signal):
        if signal is None:
            return None
        return f"\nSymbol: {signal['symbol']}\nPrice: {signal['price']}\nType: {signal['type']}\nStrength: {signal['strength']}"
