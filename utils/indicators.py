import asyncio


class Indicator:

    def __init__(self, df):
        self.df = df

    def rsi(self, period=14):
        data = self.df.head(period)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, short=12, long=26, signal=9):
        short_ema = self.df['close'].ewm(span=short, adjust=False).mean()
        long_ema = self.df['close'].ewm(span=long, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal
    
    def bollinger_bands(self, period=20, std=2):
        data = self.df.head(period)
        sma = data['Close'].rolling(window=period).mean()
        std_dev = data['Close'].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band
    
    def moving_average(self, period=10):
        data = self.df.head(period)
        return data['close'].rolling(window=period).mean()

    def calculate_atr(self, period=14):
        df = self.df
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        tr = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
