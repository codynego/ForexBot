
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
# API SETTINGS
    API_KEY = "APIKEY"
    API_SECRET = ""


    #MT5 LOGINS
    MT5_LOGIN = int(os.environ['MT5_LOGIN'])
    MT5_PASSWORD = os.environ['MT5_PASSWORD']
    MT5_SERVER = os.environ['MT5_SERVER']
    MT5_PATH = ""

    #DERIV API
    DERIV_API_TOKEN = os.environ['DERIV_API_TOKEN']

    #TELEGRAM TOKEN
    TELEGRAM_BOT_TOKEN = os.environ['TELEGRAM_BOT_TOKEN']
    TELEGRAM_CHAT_ID = os.environ['TELEGRAM_CHAT_ID']
    TELEGRAM_CHANNEL_ID = os.environ['TELEGRAM_CHANNEL_ID']
    TELEGRAM_FREE_CHANNEL_ID = os.environ['FREE_TELEGRAM_CHANNEL_ID']
    BACKTEST_ID = os.environ['BACKTEST_ID']

    #MARKETS_LIST = ["Boom 1000 Index", "Crash 1000 Index"]
    MARKETS_LIST = ["BOOM1000", "BOOM500", "CRASH1000", "CRASH500"]
    # MARKETS_LIST = ["Boom 1000 Index", "Crash 1000 Index", "Boom 500 Index", "Crash 500 Index"]
    # TIME_FRAMES = [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15]

    #TIME_FRAMES = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
    TIME_FRAMES = [900, 1800, 3600]

    CONNECTION_TIMEOUT = 3
    WEIGHTS = {"M15": 0.2, "M30": 0.7, "H1": 0.1}