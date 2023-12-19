def compute_factors(data):
    data = data.sort_values(by=['security_id', 'data_date'])
    dfs = []

    for security, df in data.groupby('security_id'):
        df = df.sort_values(by='data_date')

        # Example: Moving Average (SMA) without talib
        for i in range(6, 51, 4):
            df[f'SMA_{i}'] = df['close_price'].rolling(window=i).mean()

        # Example: Exponential Moving Average (EMA) without talib
        for i in range(6, 51, 4):
            df[f'EMA_{i}'] = df['close_price'].ewm(span=i, adjust=False).mean()

        # Example: Volume Weighted Moving Average (VWMA) without talib
        vol_price = df['volume'] * df['close_price']
        for i in range(6, 51, 4):
            df[f'VWMA_{i}'] = vol_price.rolling(window=i).mean() / df['volume'].rolling(window=i).mean()

        # Example: Bollinger Bands without talib
        for i in range(14, 61, 6):
            ma = df['close_price'].rolling(window=i).mean()
            std = df['close_price'].rolling(window=i).std()
            df[f'BBANDS_upper_{i}'] = ma + (2 * std)
            df[f'BBANDS_lower_{i}'] = ma - (2 * std)

        # Add other factors similarly using Pandas...
        
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    return final_df.dropna()
        # Momentum without talib
        for i in range(14, 61, 6):
            df[f'MOM_{i}'] = df['close_price'].diff(periods=i)

        # Acceleration - Difference in the change of momentum
        for i in range(14, 61, 6):
            df[f'ACCEL_{i}'] = df[f'MOM_{i}'].diff()

        # Rate of Change - Rate change of Price
        for i in range(14, 61, 6):
            df[f'ROCR_{i}'] = df['close_price'].pct_change(periods=i)

        # MACD - Moving Average Convergence Divergence
        for i in [18,24,30]:
            ema_fast = df['close_price'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close_price'].ewm(span=i, adjust=False).mean()
            df[f'MACD_12_{i}'] = ema_fast - ema_slow

        # RSI - Relative Strength Index
        for i in [8,14,20]:
            delta = df['close_price'].diff()
            up, down = delta.clip(lower=0), -delta.clip(upper=0)
            ema_up = up.ewm(com=i-1, adjust=False).mean()
            ema_down = down.ewm(com=i-1, adjust=False).mean()
            rs = ema_up / ema_down
            df[f'RSI_{i}'] = 100 - (100 / (1 + rs))

        # Price Volume Trend
        df['PVT'] = (df['volume'] * df['close_price'].pct_change()).cumsum()

        # OBV - On Balance Volume
        obv = (df['volume'] * (~df['close_price'].diff().le(0) * 2 - 1)).cumsum()
        df['OBV'] = obv

        # Psychological Line Indicator
        df['PSY'] = (df['close_price'] > df['close_price'].shift(1)).rolling(window=14).sum() / 14 * 100

        # Volatility - Standard Deviation of Returns
        ret = df['close_price'].pct_change()
        for i in [3, 5, 15]:
            df[f'sd_{i}'] = ret.rolling(window=i).std()

        df['sd5_15'] = df['sd_5'] / df['sd_15']

        # Moving Volume Volatility
        for i in [3, 5, 15]:
            df[f'volsd_{i}'] = df['volume'].rolling(window=i).std()

        df['volsd5_15'] = df['volsd_5'] / df['volsd_15']

        # Correlation between return and volume change
        df['vol_change'] = df['volume'].pct_change()
        for i in [5, 15]:
            df[f'corr_{i}'] = ret.rolling(i).corr(df['vol_change'])

        # Target variable
        df['target'] = df['excess_ret1d'].shift(-1)
        df['tmr_ret1d'] = df['ret1d'].shift(-1)

        dfs.append(df)
