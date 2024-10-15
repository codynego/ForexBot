import asyncio
from bot import TradingBot
from config import Config
from datetime import datetime, timedelta
from telegram import Update
import logging
from datetime import datetime, timedelta
import pytz
from telebot import send_telegram_message, start
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import utc


# # Initialize bot with credentials from config
bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)


# send signal to telegram bot

async def run_bot(api) -> None:
    try:
        print("fetching data...")
        timezone = pytz.timezone("Etc/UTC")
        end_time = datetime.now(tz=timezone)
        start_time = end_time - timedelta(minutes=4800)
            
        data = await bot.fetch_data_for_multiple_markets(api, Config.MARKETS_LIST)
            
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)
        #signals = bot.aiprocess_multiple_market(data, Config.MARKETS_LIST, signals2)
            
        for signal in signals:
            if signal is None:
                continue
            elif  signal["type"] == "HOLD":
                continue
            # else:
                # if signal['symbol'].startswith("BOOM") and signal["type"] == "SELL":
                #     continue
                # elif signal['symbol'].startswith("CRASH") and signal["type"] == "BUY":
                #     continue

            # elif signal["type"][0] == "HOLD" and signal["type"][1] == "HOLD" and signal["type"][2] == "HOLD":
            #     continue
            # else:
            #     if signal['symbol'].startswith("BOOM") and "SELL" in signal["type"]:
            #         continue
            #     elif signal['symbol'].startswith("CRASH") and "BUY" in signal["type"]:
            #         continue
            print(bot.signal_toString(signal))
            print("=============================")
            await send_telegram_message(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHANNEL_ID, bot.signal_toString(signal))
            logging.info("Signal: %s", bot.signal_toString(signal))
    except Exception as e:
        logging.error("Error: %s", str(e))






import asyncio
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pytz import utc

async def main():
    connect, api = await bot.connect_deriv(app_id="1089")
    try_count = 0
    while not connect:
        if try_count >= Config.CONNECTION_TIMEOUT:
            print("failed to connect!")
            raise Exception("Bot not initialized")

        print("Failed to initialize trading bot.")
        print("retrying in 3 seconds")
        await asyncio.sleep(3)
        try_count += 1

        print("trying to reconnect...")
        connect, api = await bot.connect_deriv(app_id="1089")

    print("bot connected")
    async def ping_api(api):
        try:
            await api.ping({"ping": 1})
        except Exception as e:
            logging.error("Ping failed: %s", str(e))
            await reconnect()

    async def reconnect():
        connect, api = await bot.connect_deriv(app_id="1089")
        while not connect:
            print("Retrying to connect...")
            await asyncio.sleep(3)
            connect, api = await bot.connect_deriv(app_id="1089")
        print("Reconnected successfully")
        await ping_api(api)

    async def run_bot_wrapper(api):
        try:
            await run_bot(api)
        except Exception as e:
            logging.error("Run bot failed: %s", str(e))

    # ping_scheduler = AsyncIOScheduler(timezone=utc)
    # ping_scheduler.add_job(ping_api, 'interval', minutes=1, args=[api])
    # ping_scheduler.start()

    scheduler = AsyncIOScheduler(timezone=utc)
    if not connect:
        await reconnect()
    else:
        scheduler.add_job(ping_api, 'interval', minutes=1, args=[api])
        scheduler.add_job(run_bot_wrapper, 'interval', minutes=15, args=[api])
        scheduler.start()

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())  
        # Run the main function using asyncio

        BOT_TOKEN = Config.TELEGRAM_BOT_TOKEN
        app = ApplicationBuilder().token(BOT_TOKEN).build()

        # Register command handlers
        app.add_handler(CommandHandler("start", start))
    except KeyboardInterrupt:
        print("Shutting down bot...")
        # Disconnect the bot on exit