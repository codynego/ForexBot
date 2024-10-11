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

async def ping_api(api):
    await api.ping({"ping": 1})


async def run_bot(api):
    try:
        print("fetching data...")
        timezone = pytz.timezone("Etc/UTC")
        end_time = datetime.now(tz=timezone)
        start_time = end_time - timedelta(minutes=4800)
            
        data = await bot.fetch_data_for_multiple_markets(api, Config.MARKETS_LIST)
            
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)
        #signals = bot.aiprocess_multiple_market(data, Config.MARKETS_LIST, signals2)
            
        for signal in signals:
            # if signal is None:
            #     continue
            # elif signal["type"] == "HOLD":
            #     continue
            # else:
            #     if signal['symbol'].startswith("BOOM") and signal['type'] == "SELL":
            #         continue
            #     elif signal['symbol'].startswith("CRASH") and signal['type'] == "BUY":
            #         continue
            print(bot.signal_toString(signal))
            print("=============================")
            await send_telegram_message(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHANNEL_ID, bot.signal_toString(signal))
            logging.info("Signal: %s", bot.signal_toString(signal))
    except Exception as e:
        logging.error("Error: %s", str(e))

async def main():
    connect, api = await bot.connect_deriv(app_id="1089")
    try_count = 0
    if await api.ping({"ping": 1}):
        print("API connected")
    while not connect:
        if try_count >= Config.CONNECTION_TIMEOUT:
            print("failed to connect!")
            raise Exception("Bot not initializeddd")

        print("Failed to initialize trading bot.")
        print("retrying in 3 seconds")
        await asyncio.sleep(3)

        print("trying to reconnect...")
        connect = bot.connect_deriv(app_id="1089")

    
        
    print("bot connecteds")

    scheduler = AsyncIOScheduler(timezone=utc)
    #scheduler.add_job(ping_api, 'interval', minutes=1, args=[api])
    scheduler.add_job(run_bot, 'interval', minutes=15, args=[api])
    scheduler.start()

    # Keep the main function running
    while True:
        await api.ping({"ping": 1})
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
