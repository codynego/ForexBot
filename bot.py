import os
import django
import asyncio
import pandas as pd

from datetime import datetime, timedelta
from asgiref.sync import sync_to_async

from config import Config
from deriv_api import DerivAPI

from tradebot.models import Market, Indicator as IndicatorModel, Signal
from utils.indicators import Indicator
from utils.strategies import Strategy

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fxbot.settings')
django.setup()


class TradingBot:
    def __init__(self, login: str, password: str, server: str):
        """
        :param login: your Deriv login
        :param password: your Deriv password
        :param server: Deriv API server endpoint
        """
        self.login = login
        self.password = password
        self.server = server

        self.connected = False
        self.signals_cache = {}
        self.prev_predictions = {}
        self.pending_signals = {}
        self.opened_positions = {}

        # Cool‑down timestamps: { symbol: datetime_of_last_signal }
        self.signal_timestamps: dict[str, datetime] = {}

    async def connect_deriv(self, app_id: str):
        """
        Authorize with the Deriv API.
        """
        api = DerivAPI(app_id=app_id)
        api_token = Config.DERIV_API_TOKEN
        authorize = await api.authorize(api_token)
        self.connected = True
        return authorize, api

    async def fetch_data_Deriv(self, api: DerivAPI, symbol: str, timeframe: int) -> pd.DataFrame | None:
        """
        Fetch candle data for a symbol at the given timeframe.
        """
        try:
            params = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 1000,
                "end": "latest",
                "style": "candles",
                "granularity": timeframe
            }
            resp = await api.ticks_history(params)
            df = pd.DataFrame(resp["candles"])
            df["datetime"] = pd.to_datetime(df["epoch"], unit="s")
            return df
        except Exception as e:
            print(f"[fetch_data] Error fetching {symbol}@{timeframe}: {e}")
            return None

    async def fetch_all_timeframes(self, api: DerivAPI, symbol: str) -> list[pd.DataFrame | None]:
        """
        Fetch data for all configured timeframes for one market.
        """
        tasks = [
            self.fetch_data_Deriv(api, symbol, tf)
            for tf in Config.TIME_FRAMES
        ]
        return await asyncio.gather(*tasks)

    async def fetch_data_for_multiple_markets(
        self,
        api: DerivAPI,
        symbols: list[str]
    ) -> list[list[pd.DataFrame | None]]:
        """
        Fetch concurrently for a list of symbols.
        """
        tasks = [
            asyncio.create_task(self.fetch_all_timeframes(api, sym))
            for sym in symbols
        ]
        return await asyncio.gather(*tasks)

    async def generate_signal(
        self,
        data: list[pd.DataFrame | None],
        strategy: str = "rsistrategy",
        symbol: str = None
    ) -> dict | None:
        """
        Generate a trading signal for `symbol`, enforcing a 30‑minute cooldown.
        Returns a dict like {"symbol":..., "price":..., "type":..., "strength":...} or None.
        """
        now = datetime.now()  # consider django.utils.timezone.now() for aware datetimes

        # 1) Cool‑down check
        last_ts = self.signal_timestamps.get(symbol)
        if last_ts and (now - last_ts) < timedelta(minutes=30):
            cooldown_end = last_ts + timedelta(minutes=30)
            print(f"[{symbol}] Cool-down active until {cooldown_end}. Skipping.")
            return None

        # 2) Build base signal
        df_latest = data[0]
        if df_latest is None or df_latest.empty:
            return None

        price = df_latest["close"].iloc[-1]
        signal = {
            "symbol": symbol,
            "price": price,
            "type": None,
            "strength": None
        }

        # 3) Strategy logic
        if strategy.lower() == "rsistrategy":
            result = await Strategy.process_multiple_timeframes(data, symbol)
            if not result:
                return None

            stra, strength, all_signals, confidence = result
            signal["strength"] = round(strength, 2)

            if stra == 1:
                signal["type"] = all_signals
            elif stra == -1:
                signal["type"] = "HOLD"
            else:
                signal["type"] = all_signals

            # filter out invalid combos
            if (symbol.startswith("BOOM") and signal["type"] == "SELL") or \
               (symbol.startswith("CRASH") and signal["type"] == "BUY"):
                return None

        # 4) Update timestamp on real signal
        if signal["type"] is not None:
            self.signal_timestamps[symbol] = now

        return signal

    async def process_multiple_signals(
        self,
        data_list: list[list[pd.DataFrame | None]],
        market_list: list[str]
    ) -> list[dict | None]:
        """
        Apply generate_signal to each (data, market) pair concurrently.
        """
        coros = [
            self.generate_signal(data, symbol=market)
            for data, market in zip(data_list, market_list)
        ]
        return await asyncio.gather(*coros, return_exceptions=False)

    def signal_toString(self, signal: dict | None) -> str | None:
        """
        Human‑readable representation of a signal.
        """
        if not signal:
            return None

        typ = signal["type"]
        entry, exit_ = None, None

        # define entry/exit rules
        if typ in (["BUY","BUY","SELL"], ["BUY","BUY","HOLD"],
                   ["SELL","SELL","BUY"], ["SELL","SELL","HOLD"]):
            entry = "immediately"
            exit_  = "after 1 spike or 30mins"
        elif typ in (["HOLD","SELL","SELL"], ["HOLD","BUY","BUY"]):
            entry = "Enter after 15mins"
            exit_  = "exit in 30 mins"

        base = (
            f"\nSymbol: {signal['symbol']}"
            f"\nPrice: {signal['price']}"
            f"\nType:  {signal['type']}"
            f"\nStrength: {signal['strength']}"
        )
        if entry and exit_:
            base += f"\nEntry: {entry}\nExit: {exit_}"
        return base

    async def take_action(self, api_obj: DerivAPI, signal: dict):
        """
        Place trades (or other actions) based on the signal.
        """
        if not signal or not signal.get("type"):
            return

        symbol = signal["symbol"]
        price  = signal["price"]
        typ    = signal["type"]

        # fetch balance (example)
        bal = await api_obj.balance({"balance": 1, "subscribe": 0})
        print(f"[take_action] Balance: {bal}")
        pass
