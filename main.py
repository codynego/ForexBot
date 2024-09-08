import asyncio
from bot import TradingBot
from config import Config
from datetime import datetime, timedelta
from telegram import Update
import logging
from datetime import datetime, timedelta
import pytz
from telebot import send_telegram_message



# # Initialize bot with credentials from config
bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)


# send signal to telegram bot

async def run_bot(api):
    try:
        print("fetching data...")
        timezone = pytz.timezone("Etc/UTC")
        end_time = datetime.now(tz=timezone)
        start_time = end_time - timedelta(minutes=4800)  # 34 hours ago
            
        data = await bot.fetch_data_for_multiple_markets(api, Config.MARKETS_LIST)
            
        signals = await bot.process_multiple_signals(data, Config.MARKETS_LIST)
        #signals = bot.aiprocess_multiple_market(data, Config.MARKETS_LIST, signals2)
            
        for signal in signals:

            if signal is None or signal["type"] == "HOLD":
                continue
            else:
                print(signal)
                await send_telegram_message("7538149095:AAHcaUUUlPVwY3q47LSouj3rY5ovNVobPE4", 5039116218, bot.signal_toString(signal))
                logging.info("Signal: %s", bot.signal_toString(signal))
    except Exception as e:
        logging.error("Error: %s", str(e))

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

        print("trying to reconnect...")
        connect = bot.connect_deriv(app_id="1089")
        
    print("bot connecteds")

    while True:
        await run_bot(api)
        await asyncio.sleep(300)

if __name__ == "__main__":
    try:
        asyncio.run(main())  # Run the main function using asyncio
    except KeyboardInterrupt:
        print("Shutting down bot...")
        # Disconnect the bot on exit
