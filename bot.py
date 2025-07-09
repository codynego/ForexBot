import pandas as pd
from utils.indicators import Indicator
from utils.strategies import Strategy
import asyncio
from config import Config
import os
import django
from asgiref.sync import sync_to_async
from deriv_api import DerivAPI
from datetime import datetime, timedelta

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fxbot.settings')
django.setup()

from tradebot.models import Market, Indicator as IndicatorModel, Signal

class TradingBot:
    def __init__(self, login, password, server):
        self.connected = False
        self.signals_cache = {}
        self.prev_predictions = {}
        self.pending_signals = {}
        self.opened_positions = {}
    
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
    
    async def fetch_data_for_multiple_markets(self, api, markets):
        data_tasks = [asyncio.create_task(self.fetch_all_timeframes(api, market)) for market in markets]
        return await asyncio.gather(*data_tasks)
            
    def apply_strategy(self, data, strategy):
        indicator = Indicator(data.head(14))
        calc = indicator.rsi()
        last_indicator_value = calc.tail(1).values[0]
        print(last_indicator_value)

    async def generate_signal(self, data, strategy="rsistrategy", symbol=None):
        # Check cooldown period for the market
        current_time = datetime.now()
        if symbol in self.signal_timestamps:
            last_signal_time = self.signal_timestamps[symbol]
            if (current_time - last_signal_time) < timedelta(minutes=30):
                print(f"Cooldown active for {symbol}. Skipping signal generation.")
                return None

        price = data[0]['close'].iloc[-1]
        signal = {"symbol": symbol, "price": price, "type": None, "strength": None}
        
        if strategy == "rsistrategy":
            result = await Strategy.process_multiple_timeframes(data, symbol)
            
            if result is None:
                return None
            stra, strength, all_signals, confidence = result
     
            signal["strength"] = round(strength, 2)
            
            if stra == 1:
                signal["type"] = all_signals
            elif stra == -1:
                signal["type"] = "HOLD"
            elif stra == 0:
                signal["type"] = all_signals

            if signal['symbol'].startswith("BOOM") and signal['type'] == "SELL":
                return None
            elif signal['symbol'].startswith("CRASH") and signal['type'] == "BUY":
                return None

            # Update timestamp cache if a valid signal is generated
            if signal["type"] is not None:
                self.signal_timestamps[symbol] = current_time
            
            return signal

    async def process_multiple_signals(self, data_list, market_list):
        signals = await asyncio.gather(*(self.generate_signal(data, symbol=market) for data, market in zip(data_list, market_list)), return_exceptions=True)
        return signals

    def signal_toString(self, signal):
        if signal is None:
            return None
        type = signal['type']
        if type == ['BUY', 'BUY', 'SELL'] or type == ['BUY', 'BUY', 'HOLD'] or type == ['SELL', 'SELL', 'BUY'] or type == ['SELL', 'SELL', 'HOLD']:
            entry = "immediately"
            exit = "after 1 spike or 30mins"
            return f"\nSymbol: {signal['symbol']}\nPrice: {signal['price']}\nType: {signal['type']}\nStrength: {signal['strength']}\nEntry: {entry}\nExit: {exit}"
        elif type == ['HOLD', 'SELL', 'SELL'] or type == ['HOLD', 'BUY', 'BUY']:
            entry = "Enter after 15mins"
            exit = "exit in 30 mins"
            return f"\nSymbol: {signal['symbol']}\nPrice: {signal['price']}\nType: {signal['type']}\nStrength: {signal['strength']}\nEntry: {entry}\nExit: {exit}"
        else:
            return f"\nSymbol: {signal['symbol']}\nPrice: {signal['price']}\nType: {signal['type']}\nStrength: {signal['strength']}"

    async def take_action(self, api_obj, signal):
        signal_type = signal["type"]
        symbol = signal["symbol"]
        price = signal["price"]
        balance = await api_obj.balance({
            "balance": 1,
            "subscribe": 0
        })
        print(balance)