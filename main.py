import asyncio
import logging
import random
from datetime import datetime, timedelta
from pytz import utc
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import ApplicationBuilder, CommandHandler
from config import Config
from bot import TradingBot
from deriv_api import DerivAPI
from telebot import send_telegram_message, start

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Initialize bot and API
bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)
connection = None
new_api = None

# Cooldown and signal tracking
FREE_SIGNAL_COUNT = 0
LAST_RESET_TIME = datetime.now(utc)
last_telegram_sent = {}  # {symbol: datetime}

async def send_message(token, message, symbol):
    global FREE_SIGNAL_COUNT, LAST_RESET_TIME, last_telegram_sent

    now = datetime.now(utc)

    # Reset daily free signal limit every 24 hours
    if now - LAST_RESET_TIME >= timedelta(hours=24):
        FREE_SIGNAL_COUNT = 0
        LAST_RESET_TIME = now

    # Check 30-minute cooldown for this symbol
    last_sent = last_telegram_sent.get(symbol)
    if last_sent and (now - last_sent) < timedelta(minutes=30):
        cooldown_end = last_sent + timedelta(minutes=30)
        logging.info(f"[{symbol}] Cooldown active until {cooldown_end}, skipping Telegram.")
        return

    try:
        # Send to premium channel
        await send_telegram_message(token, Config.TELEGRAM_CHANNEL_ID, message)

        # Send to free channel (limited to 3 signals/day, 25% chance)
        if FREE_SIGNAL_COUNT < 3 and random.choices([True, False], weights=[1, 3])[0]:
            await send_telegram_message(token, Config.TELEGRAM_FREE_CHANNEL_ID, message)
            FREE_SIGNAL_COUNT += 1

        last_telegram_sent[symbol] = now  # Update cooldown tracker
        logging.info(f"Sent signal for {symbol}: {message}")

    except Exception as e:
        logging.error(f"Error sending Telegram message: {str(e)}")

async def reconnect():
    global connection, new_api
    for attempt in range(5):
        try:
            connection, new_api = await bot.connect_deriv(app_id="1089")
            if connection:
                ping = await new_api.ping({"ping": 1})
                if ping.get("ping"):
                    logging.info("Deriv API reconnected successfully.")
                    return True
        except Exception as e:
            logging.error(f"Reconnect attempt {attempt + 1} failed: {e}")
        await asyncio.sleep(10)
    logging.error("Max reconnect attempts reached.")
    return False

async def run_bot():
    global connection, new_api

    try:
        if not new_api:
            logging.warning("API not connected. Attempting to reconnect...")
            if not await reconnect():
                return

        logging.info("Fetching market data...")
        data = await bot.fetch_data_for_multiple_markets(new_api, Config.MARKETS_LIST)
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)

        for signal in signals:
            if signal is None:
                continue

            symbol = signal["symbol"]
            signal_type = signal["type"]

            # Filter invalid signals
            if symbol.startswith("BOOM") and signal_type.count("SELL") > 1:
                continue
            if symbol.startswith("CRASH") and signal_type.count("BUY") > 1:
                continue
            if signal_type.count("HOLD") > 1:
                continue
            if signal_type.count("BUY") == 1 and signal_type.count("SELL") == 1:
                continue
            if signal_type == "HOLD":
                continue

            signal_text = bot.signal_toString(signal)
            print(signal_text)
            print("============================")

            # Send signal with cooldown check
            await send_message(Config.TELEGRAM_BOT_TOKEN, signal_text, symbol)

    except Exception as e:
        logging.error(f"run_bot failed: {str(e)}")
        if await reconnect():
            await run_bot()
        else:
            logging.error("run_bot retry failed.")

async def ping_api():
    global connection, new_api
    try:
        if new_api:
            ping = await new_api.ping({"ping": 1})
            logging.info(f"Ping successful: {ping['ping']}")
        else:
            logging.warning("API is None, ping skipped.")
            await reconnect()
    except Exception as e:
        logging.error(f"Ping error: {str(e)}")
        await reconnect()

async def schedule_jobs():
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.add_job(ping_api, "interval", minutes=1)
    scheduler.add_job(run_bot, "interval", minutes=2)
    scheduler.start()

async def main():
    global connection, new_api

    connection, new_api = await bot.connect_deriv(app_id="1089")

    if not connection:
        logging.warning("Initial connection failed. Attempting to reconnect...")
        await reconnect()

    if connection:
        logging.info("Bot connected to Deriv API.")
        await new_api.ping({"ping": 1})
        await schedule_jobs()
    else:
        logging.error("Failed to establish connection.")

    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())

        # Telegram bot setup for interactivity
        app = ApplicationBuilder().token(Config.TELEGRAM_BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.run_polling()
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")
        print("Bot shutdown requested.")