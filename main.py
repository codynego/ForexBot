import asyncio
import logging
import random
from contextlib import suppress
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pytz import utc

from bot import TradingBot
from config import Config
from telebot import send_telegram_message


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


bot = TradingBot(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)
new_api = None

FREE_SIGNAL_COUNT = 0
LAST_RESET_TIME = datetime.now(utc)
last_telegram_sent = {}
last_api_activity = None

# The Deriv websocket client is shared across jobs, so we guard it to avoid
# concurrent requests leaving the connection in a bad state.
api_lock = asyncio.Lock()
run_lock = asyncio.Lock()


async def close_api():
    global new_api, last_api_activity
    if not new_api:
        return

    for method_name in ("disconnect", "close", "clear"):
        method = getattr(new_api, method_name, None)
        if not method:
            continue

        try:
            result = method()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logging.exception("Failed while closing Deriv API with %s", method_name)
        finally:
            break

    new_api = None
    last_api_activity = None


async def call_api(operation_name, coroutine_factory, timeout):
    global last_api_activity

    async with api_lock:
        if new_api is None:
            raise ConnectionError("Deriv API client is not connected.")

        try:
            result = await asyncio.wait_for(coroutine_factory(new_api), timeout=timeout)
            last_api_activity = datetime.now(utc)
            return result
        except asyncio.TimeoutError:
            logging.error("%s timed out after %ss. Recycling Deriv connection.", operation_name, timeout)
            await close_api()
            raise
        except asyncio.CancelledError:
            logging.error("%s was cancelled. Recycling Deriv connection.", operation_name)
            await close_api()
            raise
        except Exception:
            logging.exception("%s failed. Recycling Deriv connection.", operation_name)
            await close_api()
            raise


async def send_message(token, message, symbol):
    global FREE_SIGNAL_COUNT, LAST_RESET_TIME, last_telegram_sent

    now = datetime.now(utc)
    if now - LAST_RESET_TIME >= timedelta(hours=24):
        FREE_SIGNAL_COUNT = 0
        LAST_RESET_TIME = now

    last_sent = last_telegram_sent.get(symbol)
    if last_sent and (now - last_sent) < timedelta(minutes=30):
        cooldown_end = last_sent + timedelta(minutes=30)
        logging.info("[%s] Cooldown active until %s, skipping Telegram.", symbol, cooldown_end)
        return

    try:
        await send_telegram_message(token, Config.TELEGRAM_CHANNEL_ID, message)

        if FREE_SIGNAL_COUNT < 3 and random.choices([True, False], weights=[1, 3])[0]:
            await send_telegram_message(token, Config.TELEGRAM_FREE_CHANNEL_ID, message)
            FREE_SIGNAL_COUNT += 1

        last_telegram_sent[symbol] = now
        logging.info("Sent signal for %s", symbol)
    except Exception:
        logging.exception("Error sending Telegram message for %s", symbol)


async def reconnect():
    global new_api, last_api_activity

    async with api_lock:
        await close_api()

        for attempt in range(1, 6):
            candidate_api = None
            try:
                _, candidate_api = await bot.connect_deriv(app_id="1089")
                new_api = candidate_api
                last_api_activity = datetime.now(utc)
                logging.info("Deriv API connected on attempt %s.", attempt)
                return True
            except Exception:
                logging.exception("Reconnect attempt %s failed", attempt)
                with suppress(Exception):
                    close_result = getattr(candidate_api, "disconnect", None)
                    if close_result:
                        result = close_result()
                        if asyncio.iscoroutine(result):
                            await result
            await asyncio.sleep(5)

    logging.error("Max reconnect attempts reached.")
    return False


async def ensure_connection():
    if new_api is not None:
        return True
    logging.warning("API not connected. Attempting reconnect.")
    return await reconnect()


def is_actionable_signal(signal):
    if signal is None:
        return False

    signal_type = signal.get("type")
    if signal_type == "HOLD":
        return False
    if not isinstance(signal_type, list):
        return False
    if signal_type.count("HOLD") > 1:
        return False
    if signal_type.count("BUY") == 1 and signal_type.count("SELL") == 1:
        return False

    symbol = signal.get("symbol", "")
    if symbol.startswith("BOOM") and signal_type.count("SELL") > 1:
        return False
    if symbol.startswith("CRASH") and signal_type.count("BUY") > 1:
        return False
    return True


async def run_bot():
    if run_lock.locked():
        logging.warning("Previous run is still active; skipping overlapping cycle.")
        return

    async with run_lock:
        if not await ensure_connection():
            return

        try:
            logging.info("Fetching market data...")
            data = await call_api(
                "fetch_data_for_multiple_markets",
                lambda api: bot.fetch_data_for_multiple_markets(api, Config.MARKETS_LIST),
                timeout=60,
            )

            if not data or len(data) != len(Config.MARKETS_LIST):
                logging.error("Received incomplete market data; skipping this cycle.")
                await reconnect()
                return

            valid_pairs = []
            for market, market_data in zip(Config.MARKETS_LIST, data):
                if not market_data or any(frame is None or frame.empty for frame in market_data):
                    logging.warning("Missing candles for %s; skipping it this cycle.", market)
                    continue
                valid_pairs.append((market_data, market))

            if not valid_pairs:
                logging.error("No valid market data was available.")
                await reconnect()
                return

            signals = await bot.process_multiple_signals(
                [market_data for market_data, _ in valid_pairs],
                [market for _, market in valid_pairs],
            )

            for market, signal in zip([market for _, market in valid_pairs], signals):
                if isinstance(signal, Exception):
                    logging.error(
                        "Signal generation failed for %s",
                        market,
                        exc_info=(type(signal), signal, signal.__traceback__),
                    )
                    continue
                if not is_actionable_signal(signal):
                    continue

                signal_text = bot.signal_toString(signal)
                if not signal_text:
                    continue

                logging.info("Actionable signal generated for %s", market)
                print(signal_text)
                print("============================")
                await send_message(Config.TELEGRAM_BOT_TOKEN, signal_text, market)

        except asyncio.TimeoutError:
            logging.exception("run_bot timed out; reconnecting.")
            await reconnect()
        except Exception:
            logging.exception("run_bot failed unexpectedly")
            await reconnect()


async def ping_api():
    if not await ensure_connection():
        return

    try:
        if run_lock.locked():
            logging.info("Skipping ping while signal cycle is running.")
            return

        if last_api_activity and (datetime.now(utc) - last_api_activity) < timedelta(seconds=90):
            logging.info("Skipping ping because the Deriv connection was recently active.")
            return

        ping = await call_api(
            "ping",
            lambda api: api.ping({"ping": 1}),
            timeout=20,
        )
        logging.info("Ping successful: %s", ping.get("ping"))
    except Exception:
        logging.exception("Ping error")
        await reconnect()


async def schedule_jobs():
    scheduler = AsyncIOScheduler(timezone=utc)
    scheduler.add_job(
        ping_api,
        trigger=IntervalTrigger(minutes=1),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    scheduler.add_job(
        run_bot,
        trigger=IntervalTrigger(minutes=2),
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )
    scheduler.start()
    return scheduler


async def main():
    scheduler = None
    try:
        if not await reconnect():
            logging.error("Failed to establish initial Deriv connection.")
            return

        logging.info("Bot connected to Deriv API.")
        scheduler = await schedule_jobs()
        await run_bot()

        while True:
            await asyncio.sleep(60)
    except KeyboardInterrupt:
        logging.info("Bot stopped manually.")
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)
        await close_api()


if __name__ == "__main__":
    asyncio.run(main())
