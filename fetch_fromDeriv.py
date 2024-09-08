from deriv_api import DerivAPI
import asyncio
from config import Config
import pandas as pd


async def fecth_data(app_id):
    api = DerivAPI(app_id=app_id)
    api_token = Config.DERIV_API_TOKEN

    authorize = await api.authorize(api_token)
    if authorize:
        print('Authorized')
        # get tick history
        ticks = await api.ticks_history({
            "ticks_history": "BOOM500",
            "adjust_start_time": 1,
            "count": 10,
            "end": "latest",
            "start": 1,
            "style": "candles",
            }
        )
        df = pd.DataFrame(list(ticks['candles']))
        df['datetime'] = pd.to_datetime(df['epoch'], unit='s')
        print(df)

async def main():
    app_id = '1089'
    await fecth_data(app_id)

if __name__ == "__main__":
    asyncio.run(main())  # Run the main function using asyncio