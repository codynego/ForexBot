import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
import asyncio

# Replace with your actual Telegram bot token
BOT_TOKEN = "7538149095:AAHcaUUUlPVwY3q47LSouj3rY5ovNVobPE4"

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize an in-memory list to store chat IDs (in a real-world application, use a database)
user_chat_ids = set()


async def start(update: Update, context: CallbackContext):
    """Greets the user and stores their chat ID."""
    chat_id = update.message.chat_id
    user_chat_ids.add(chat_id)  # Store the user's chat ID

    await update.message.reply_text(
        'Hi! You have been added to the broadcast list. Use /broadcast [message] to send a message to all users.'
    )


async def echo(update: Update, context: CallbackContext):
    """Echoes the user message and stores the chat ID."""
    chat_id = update.message.chat_id
    user_chat_ids.add(chat_id)  # Store the user's chat ID

    await update.message.reply_text(update.message.text)


async def broadcast(update: Update, context: CallbackContext):
    """Broadcasts the given message to all stored user chat IDs."""
    # Extract the message to broadcast
    message_text = update.message.text[len('/broadcast '):].strip()
    if not message_text:
        await update.message.reply_text("Please provide a message to broadcast.")
        return

    # Broadcast the message to all users in the chat ID list
    failed = 0
    for chat_id in user_chat_ids:
        try:
            await context.bot.send_message(chat_id, message_text)
        except Exception as e:
            logger.error(f"Failed to send message to {chat_id}: {e}")
            failed += 1

    success_count = len(user_chat_ids) - failed
    await update.message.reply_text(f"Message broadcasted to {success_count} users. Failed to send to {failed} users.")


async def main():
    """Creates the Application, registers handlers, and starts the bot."""
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("broadcast", broadcast))

    # Add a general message handler to store chat IDs on any message
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start the bot
    app.run_polling()


if __name__ == '__main__':
    """Creates the Application, registers handlers, and starts the bot."""
    
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("broadcast", broadcast))

    # Add a general message handler to store chat IDs on any message
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    # Start the bot
    app.run_polling()