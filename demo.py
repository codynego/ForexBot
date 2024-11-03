import asyncio
import logging
from datetime import datetime, timedelta
from django.db import connection
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

connection = None
new_api = None


async def run_bot(new_api, connect) -> None:
    response = await new_api.ping({"ping": 1})
    if not response['ping']:
        logging.error("API not connected. Cannot run bot.")
        reconnect_result = await reconnect()
        if reconnect_result is not None:
            connection, new_api = reconnect_result
            response = await new_api.ping({"ping": 1})
            if response['ping']:
                print("API connected successfully.")
                await run_bot(new_api, connect)
            else:
                logging.error("API not connected. Cannot run bot.")
                return
            response = await new_api.ping({"ping": 1})
            if response['ping']:
                print("API connected successfully.")
                await run_bot(new_api, connect)
            else:
                logging.error("API not connected. Cannot run bot.")
                return
            await run_bot(new_api, connect)
    if new_api is None or not connect:
        logging.error("API not connected. Cannot run bot.")
        reconnect_result = await reconnect()
        if reconnect_result is not None:
            connection, new_api = reconnect_result
    try:
        print("Fetching data...")
        timezone = pytz.timezone("Etc/UTC")
        end_time = datetime.now(tz=timezone)
        start_time = end_time - timedelta(minutes=4800)
        
        # Fetch market data
        data = await bot.fetch_data_for_multiple_markets(new_api, Config.MARKETS_LIST)
        
        # Process multiple signals
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)
        
        for signal in signals:
            if signal is None or signal["type"] == "HOLD":
                continue

            
            # # #Skip unwanted signals based on market type
            # if signal['symbol'].startswith("BOOM") and signal["type"].count("SELL") > 1:
            #     continue
            # elif signal['symbol'].startswith("CRASH") and signal["type"].count("BUY") > 1:
            #     continue
            # elif signal["type"].count("HOLD") > 1:
            #     continue
            # elif signal["type"].count("BUY") == 1 and signal["type"].count("SELL") == 1:
            #     continue

            # Send signal to Telegram
            print(bot.signal_toString(signal))
            print("============================")
            #await send_message(Config.TELEGRAM_BOT_TOKEN, bot.signal_toString(signal))
            #await send_telegram_message(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHANNEL_ID, bot.signal_toString(signal))
            logging.info("Signal: %s", bot.signal_toString(signal))
    except Exception as e:
        logging.error("Run bot failed: %s", str(e))


# Global variables to keep track of free signal count and reset time
FREE_SIGNAL_COUNT = 0
LAST_RESET_TIME = datetime.now()

async def send_message(token, message):
    global FREE_SIGNAL_COUNT, LAST_RESET_TIME

    free_channel = Config.TELEGRAM_FREE_CHANNEL_ID
    premium_channel = Config.TELEGRAM_CHANNEL_ID

    # Check if 24 hours have passed since the last reset
    if datetime.now() - LAST_RESET_TIME >= timedelta(hours=24):
        FREE_SIGNAL_COUNT = 0
        LAST_RESET_TIME = datetime.now()

    try:
        await send_telegram_message(token, premium_channel, message)
        if FREE_SIGNAL_COUNT < 3 and random.choices([True, False], weights=[1, 3])[0]:
            await send_telegram_message(token, free_channel, message)
            FREE_SIGNAL_COUNT += 1
    except Exception as e:
        logging.error("Error sending message to telegram: %s", str(e))



async def run_bot_wrapper(new_api, connect):
    try:
        await run_bot(new_api, connect)
    except Exception as e:
        logging.error("Run bot failed: %s", str(e))
        reconnect_result = await reconnect()
        if reconnect_result is not None:
            logging.info("Reconnected successfully after run bot failure.")
            await run_bot_wrapper(new_api, connection)

        else:
            logging.error("Reconnection failed after run bot failure.")
            return
            
            



async def ping_api(new_api):
    try:
        if new_api is not None:
            response = await new_api.ping({"ping": 1})
            if response['ping']:
                print(response['ping'])
        else:
            logging.error("new_api is None, cannot ping.")
    except Exception as e:
        logging.error("Ping failed: %s. Attempting to reconnect...", str(e))
        reconnect_result = await reconnect()
        if reconnect_result is not None:
            connection, new_api = reconnect_result
            logging.info("Reconnected successfully after ping failure.")
            await ping_api(new_api)
        else:
            logging.error("Reconnection failed after ping failure.")
            return      
async def reconnect():
    retry_attempts = 0
    max_retries = 5
    connection, new_api = await bot.connect_deriv(app_id="1089")
    response = await new_api.ping({"ping": 1})

    while not connection and retry_attempts < max_retries:
        print(f"Retrying to connect... attempt {retry_attempts + 1}")
        # await asyncio.sleep(120)
        retry_attempts += 1
        connect, api = await bot.connect_deriv(app_id="1089")
    
        if retry_attempts >= max_retries:
            logging.error("Max retries exceeded. Could not reconnect.")
            return
        
    print("Reconnected successfully")
    await new_api.ping({"ping": 1})
    return connection, new_api


async def schedule_jobs(new_api, connection):
    try:
        scheduler = AsyncIOScheduler(timezone=utc)
    
        scheduler.add_job(ping_api, 'interval', minutes=1, args=[new_api])
        
        # Schedule bot to run every 15 minutes
        scheduler.add_job(run_bot_wrapper, 'interval', minutes=1, args=[new_api, connection])
        
        scheduler.start()
    except Exception as e:
        logging.error("Scheduler error: %s. Attempting to reconnect...", str(e))
        
async def main():
    connection, new_api = await bot.connect_deriv(app_id="1089")

    while not connection:
        await asyncio.sleep(120)
        result = await reconnect()
        if result is not None:
            logging.error("Reconnection failed.")
            if result:
                logging.info("Reconnected successfully.")
                connection, new_api = result
                break
    print("bot connected")

    try:
        await new_api.ping({"ping": 1})
        await schedule_jobs(new_api, connection)
    except Exception as e:
        logging.error("Ping failed: %s. Attempting to reconnect...", str(e))
        reconnect_result = await reconnect()
        if reconnect_result is not None:
            connection, new_api = reconnect_result
            logging.info("Reconnected successfully after ping failure.")
            await schedule_jobs(new_api, connection)

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        # Run the main function using asyncio
        asyncio.run(main())

        # Telegram Bot Setup

        BOT_TOKEN = Config.TELEGRAM_BOT_TOKEN
        app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Register command handlers for the Telegram bot
        app.add_handler(CommandHandler("start", start))

    except KeyboardInterrupt:
        print("Shutting down bot...")
        logging.info("Bot stopped manually.")
