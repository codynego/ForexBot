import telegram
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import asyncio
from config import Config

# grap the users chat id and save it on start bot to send messages to the user
# save the chat id in the database

# Initialize an in-memory list to store chat IDs (in a real-world application, use a database)
user_chat_ids = set()


async def start(update: Update, context: CallbackContext):
    """Greets the user and stores their chat ID."""
    if update.message is not None:
        chat_id = update.message.chat_id
        await update.message.reply_text(
            'Hi! You have been added to the broadcast list. Use /broadcast [message] to send a message to all users.'
        )
    else:
        # Handle the case when there is no message associated with the update
        # You can choose to raise an exception, log an error, or handle it in any other way that makes sense for your application
        chat_id = None  # Set chat_id to a default value or handle it accordingly
    user_chat_ids.add(chat_id)  # Store the user's chat ID

async def send_telegram_message(bot_token, channel_id, signal):
    bot = telegram.Bot(token=bot_token)
    text = f"New signal received: {signal}"
    await bot.send_message(chat_id=channel_id, text=text)

from telegram import Bot
async def main():
    channel_id = -1002265016206

    bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
    await bot.send_message(chat_id=channel_id, text="hello")
    updates = await bot.get_updates()
    
    user_chat_ids = set()
    for update in updates:
        if update.message:
            user_chat_ids.add(update.message.chat_id)
    
    print("Bot Subscribers:")
    for chat_id in user_chat_ids:
        print(f"Chat ID: {chat_id}")
# print bot users

if __name__ == "__main__":
    asyncio.run(main())