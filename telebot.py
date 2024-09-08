import telegram
import asyncio

async def send_telegram_message(bot_token, channel_id, signal):
    bot = telegram.Bot(token=bot_token)
    text = f"New signal received: {signal}"
    await bot.send_message(chat_id=channel_id, text=text)


async def main():
    await send_telegram_message("",3, "testing this bot")
    print("Sent message")

if __name__ == "__main__":
    asyncio.run(main())