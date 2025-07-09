import asyncio
import logging
import random
from datetime import datetime, timedelta
from pytz import utc

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import ApplicationBuilder, CommandHandler

from config import Config
from bot import TradingBot
from telebot import send_telegram_message, start
from deriv_api import DerivAPI

# Setup
bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)
connection = None
new_api = None

# Cooldown and tracking
FREE_SIGNAL_COUNT = 0
LAST_RESET_TIME = datetime.now()
last_telegram_sent = {}  # {symbol: datetime}


async def send_message(token, message: str, symbol: str):
    """Send signal message to Telegram with cooldown and free channel logic."""
    global FREE_SIGNAL_COUNT, LAST_RESET_TIME, last_telegram_sent

    now = datetime.now()

    # Reset free signal counter every 24 hours
    if now - LAST_RESET_TIME >= timedelta(hours=24):
        FREE_SIGNAL_COUNT = 0
        LAST_RESET_TIME = now

    # Enforce 30-minute cooldown per symbol
    last_sent = last_telegram_sent.get(symbol)
    if last_sent and (now - last_sent) < timedelta(minutes=30):
        cooldown_end = last_sent + timedelta(minutes=30)
        print(f"[{symbol}] Cooldown active until {cooldown_end}, skipping Telegram.")
        return

    try:
        await send_telegram_message(token, Config.TELEGRAM_CHANNEL_ID, message)

        # Send to free channel occasionally
        if FREE_SIGNAL_COUNT < 3 and random.choices([True, False], weights=[1, 3])[0]:
            await send_telegram_message(token, Config.TELEGRAM_FREE_CHANNEL_ID, message)
            FREE_SIGNAL_COUNT += 1

        last_telegram_sent[symbol] = now
        logging.info(f"Telegram sent for {symbol}: {message}")

    except Exception as e:
        logging.error(f"Telegram send error: {e}")


async def reconnect():
    """Reconnect to Deriv API if needed."""
    global connection, new_api
    for attempt in range(5):
        try:
            connection, new_api = await bot.connect_deriv(app_id="1089")
            if connection:
                ping = await new_api.ping({"ping": 1})
                if ping.get("ping"):
                    logging.info("Deriv API reconnected.")
                    return
        except Exception as e:
            logging.error(f"Reconnect attempt {attempt + 1} failed: {e}")
        await asyncio.sleep(10)

    logging.error("Max reconnect attempts reached.")


async def run_bot():
    """Fetch market data and process trading signals."""
    global connection, new_api

    try:
        if not new_api:
            logging.warning("No active API connection. Attempting reconnect...")
            await reconnect()

        if not new_api:
            logging.error("API connection unavailable. Skipping run.")
            return

        print("Fetching data...")
        data = await bot.fetch_data_for_multiple_markets(new_api, Config.MARKETS_LIST)
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)

        for signal in signals:
            # Safeguard against bad or incomplete signal data
            if not isinstance(signal, dict) or "symbol" not in signal or "type" not in signal:
                logging.warning(f"Invalid signal skipped: {signal}")
                continue

            symbol = signal["symbol"]
            signal_type = signal["type"]

            # Filter out weak/noise signals
            if (
                (symbol.startswith("BOOM") and signal_type.count("SELL") > 1)
                or (symbol.startswith("CRASH") and signal_type.count("BUY") > 1)
                or (signal_type.count("HOLD") > 1)
                or (signal_type.count("BUY") == 1 and signal_type.count("SELL") == 1)
                or signal_type == "HOLD"
            ):
                continue

            signal_text = bot.signal_toString(signal)
            print(signal_text)
            print("=" * 30)

            await send_message(Config.TELEGRAM_BOT_TOKEN, signal_text, symbol)

    except Exception as e:
        logging.error(f"run_bot failed: {e}")
        await reconnect()
        if connection:
            await run_bot()
        else:
            logging.error("Retry run_bot failed.")


async def ping_api():
    """Ping the Deriv API to keep it alive."""
    global connection, new_api
    try:
        if new_api:
            ping = await new_api.ping({"ping": 1})
            print("Ping:", ping.get("ping"))
        else:
            logging.warning("API is None. Reconnecting...")
            await reconnect()
    except Exception as e:
        logging.error(f"Ping error: {e}")
        await reconnect()


async def schedule_jobs():
    """Schedule bot and ping jobs using APScheduler."""
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.add_job(ping_api, "interval", minutes=1)
    scheduler.add_job(run_bot, "interval", minutes=2)
    scheduler.start()


async def main():
    """Initialize connections and start scheduled tasks."""
    global connection, new_api

    connection, new_api = await bot.connect_deriv(app_id="1089")
    if not connection:
        await reconnect()

    if connection:
        print("Bot connected.")
        await new_api.ping({"ping": 1})
        await schedule_jobs()

    # Keep running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        # Start bot
        asyncio.run(main())

        # Optional Telegram bot interface
        BOT_TOKEN = Config.TELEGRAM_BOT_TOKEN
        app = ApplicationBuilder().token(BOT_TOKEN).build()
        app.add_handler(CommandHandler("start", start))

    except KeyboardInterrupt:
        print("Bot shutdown requested.")
        logging.info("Bot stopped manually.")
