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

# Initialize bot with credentials from config
bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)

async def run_bot(api) -> None:
    try:
        print("Fetching data...")
        timezone = pytz.timezone("Etc/UTC")
        end_time = datetime.now(tz=timezone)
        start_time = end_time - timedelta(minutes=4800)
        
        # Fetch market data
        data = await bot.fetch_data_for_multiple_markets(api, Config.MARKETS_LIST)
        
        # Process multiple signals
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)
        
        for signal in signals:
            if signal is None or signal["type"] == "HOLD":
                continue

            # Skip unwanted signals based on market type
            if (signal['symbol'].startswith("BOOM") and "SELL" in signal["type"]) or \
               (signal['symbol'].startswith("CRASH") and "BUY" in signal["type"]):
                continue

            # Send signal to Telegram
            print(bot.signal_toString(signal))
            print("=============================")
            #await send_telegram_message(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHANNEL_ID, bot.signal_toString(signal))
            logging.info("Signal: %s", bot.signal_toString(signal))
            
    except Exception as e:
        logging.error("Error in run_bot: %s", str(e))


async def ping_api(api):
    try:
        await api.ping({"ping": 1})
    except Exception as e:
        logging.error("Ping failed: %s", str(e))
        await reconnect()


async def reconnect():
    retry_attempts = 0
    max_retries = 5  # Retry connection up to 5 times
    connect, api = await bot.connect_deriv(app_id="1089")
    
    while not connect and retry_attempts < max_retries:
        print(f"Retrying to connect... attempt {retry_attempts + 1}")
        await asyncio.sleep(3)
        retry_attempts += 1
        connect, api = await bot.connect_deriv(app_id="1089")
    
    if retry_attempts >= max_retries:
        logging.error("Max retries exceeded. Could not reconnect.")
        return
    
    print("Reconnected successfully")
    await ping_api(api)
    return connect, api


async def run_bot_wrapper(api):
    try:
        await run_bot(api)
    except Exception as e:
        logging.error("Run bot failed: %s", str(e))


async def main():
    connect, api = await bot.connect_deriv(app_id="1089")
    try_count = 0

    # Retry connecting if failed initially
    while not connect:
        if try_count >= Config.CONNECTION_TIMEOUT:
            print("Failed to connect! Exceeded retry attempts.")
            raise Exception("Bot not initialized")

        print("Failed to initialize trading bot. Retrying in 3 seconds...")
        await asyncio.sleep(3)
        try_count += 1
        connect, api = await bot.connect_deriv(app_id="1089")

    print("Bot connected successfully!")

    # Create an AsyncIO scheduler for periodic tasks
    scheduler = AsyncIOScheduler(timezone=utc)
    
    # Schedule pings every 30 seconds
    scheduler.add_job(ping_api, 'interval', seconds=30, args=[api])
    
    # Schedule bot to run every 15 minutes
    scheduler.add_job(run_bot_wrapper, 'interval', minutes=15, args=[api])
    
    scheduler.start()

    # Keep the bot running
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
