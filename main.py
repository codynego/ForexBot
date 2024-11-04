import asyncio
import logging
from datetime import datetime, timedelta
import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pytz import utc
from telegram.ext import ApplicationBuilder, CommandHandler
from bot import TradingBot
from config import Config
from telebot import send_telegram_message, start
import random

# Initialize bot with credentials from config
bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)

# Global variables for connection and API
connection = None
new_api = None
FREE_SIGNAL_COUNT = 0
LAST_RESET_TIME = datetime.now()

async def send_message(token, message):
    global FREE_SIGNAL_COUNT, LAST_RESET_TIME

    free_channel = Config.TELEGRAM_FREE_CHANNEL_ID
    premium_channel = Config.TELEGRAM_CHANNEL_ID
    backtest = Config.BACKTEST_ID

    # Check if 24 hours have passed since the last reset
    if datetime.now() - LAST_RESET_TIME >= timedelta(hours=24):
        FREE_SIGNAL_COUNT = 0
        LAST_RESET_TIME = datetime.now()

    try:
        await send_telegram_message(token, backtest, message)
        #await send_telegram_message(token, premium_channel, message)
        if FREE_SIGNAL_COUNT < 3 and random.choices([True, False], weights=[1, 3])[0]:
            #await send_telegram_message(token, free_channel, message)
            FREE_SIGNAL_COUNT += 1
    except Exception as e:
        logging.error("Error sending message to telegram: %s", str(e))



async def reconnect():
    global connection, new_api
    retry_attempts = 0
    max_retries = 5

    while retry_attempts < max_retries:
        try:
            connection, new_api = await bot.connect_deriv(app_id="1089")
            if connection:
                response = await new_api.ping({"ping": 1})
                if response['ping']:
                    logging.info("Reconnected successfully.")
                    return
        except Exception as e:
            logging.error(f"Reconnect attempt {retry_attempts + 1} failed: {e}")
            retry_attempts += 1
            await asyncio.sleep(10)

    logging.error("Max retries exceeded. Could not reconnect.")
    return

async def run_bot():
    global connection, new_api
    try:
        if not new_api:
            logging.error("API not connected. Attempting to reconnect...")
            await reconnect()
        if new_api:
            print("Fetching data...")
            data = await bot.fetch_data_for_multiple_markets(new_api, Config.MARKETS_LIST)
            signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)
            # print(signals)
            for signal in signals:
                if signal is None:
                    continue
                #Skip unwanted signals based on market type
                if signal['symbol'].startswith("BOOM") and signal["type"].count("SELL") > 1: # type: ignore
                    continue
                elif signal['symbol'].startswith("CRASH") and signal["type"].count("BUY") > 1: # type: ignore
                    continue
                elif signal["type"].count("HOLD") > 1: # type: ignore
                    continue
                elif signal["type"].count("BUY") == 1 and signal["type"].count("SELL") == 1: # type: ignore
                    continue
                elif signal["type"] == "HOLD": # type: ignore
                    continue
                

                # Send signal to Telegram
                print(bot.signal_toString(signal))
                print("============================")
                #await send_message(Config.TELEGRAM_BOT_TOKEN, bot.signal_toString(signal))
                #await send_telegram_message(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHANNEL_ID, bot.signal_toString(signal))
                
                logging.info("Signal: %s", bot.signal_toString(signal))
    except Exception as e:
        logging.error("Run bot failed: %s", str(e))
        await reconnect()
        if connection:
            await run_bot()
        else:
            logging.error("failed to run bot: %s", str(e))
            return

async def ping_api():
    global connection, new_api
    try:
        if new_api is not None:
            response = await new_api.ping({"ping": 1})
            if response['ping']:
                print(response['ping'])
        else:
            logging.error("new_api is None, cannot ping.")
    except Exception as e:
        logging.error("Ping error: %s. Attempting to reconnect...", str(e))
        await reconnect()
        if connection:
            await ping_api()

async def schedule_jobs():
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.add_job(ping_api, 'interval', minutes=1)
    scheduler.add_job(run_bot, 'interval', minutes=15)
    scheduler.start()

async def main():
    global connection, new_api
    connection, new_api = await bot.connect_deriv(app_id="1089")

    if not connection:
        await reconnect()

    if connection:
        print("Bot connected")
        await new_api.ping({"ping": 1})
        await schedule_jobs()

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())

        # Telegram Bot Setup
        BOT_TOKEN = Config.TELEGRAM_BOT_TOKEN
        app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Register command handlers for the Telegram bot
        app.add_handler(CommandHandler("start", start))

    except KeyboardInterrupt:
        print("Shutting down bot...")
        logging.info("Bot stopped manually.")