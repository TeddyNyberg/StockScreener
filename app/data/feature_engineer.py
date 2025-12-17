#feat_engr
import pandas as pd
import numpy as np


def feat_engr(list_of_df):
    cleaned_dfs = []

    for df in list_of_df:
        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        data['Range'] = (data['High'] - data['Low']) / data['Open'].replace(0, np.nan)
        data['Delta_Percent'] = data['Close'].pct_change()
        data['Large_Move'] = ((data['Delta_Percent'] >= 0.05) | (data['Delta_Percent'] <= -0.05)).astype(float)
        data['Trend_Up'] = (data['Delta_Percent'].rolling(window=5).apply(lambda x: (x > 0).all(), raw=True).fillna(0)
                            .astype(float))
        data['Trend_Down'] = (data['Delta_Percent'].rolling(window=5).apply(lambda x: (x < 0).all(), raw=True).fillna(0)
                            .astype(float))
        data['MA_50'] = data["Close"].rolling(window=50).mean() #Yes norm
        data['Break_Up'] = np.where(data["Close"] > data['MA_50'], 1.0, 0.0)
        data['Break_Down'] = np.where(data["Close"] < data['MA_50'], 1.0, 0.0)
        std_20 = data['Close'].rolling(window=20).std()
        ma_20 = data['Close'].rolling(window=20).mean()
        data['BB_Upper'] = ma_20 + (std_20 * 2) #Yes norm
        data['BB_Lower'] = ma_20 - (std_20 * 2) #Yes norm

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'] = (data['RSI'].fillna(50))/100
        data['Overbought_RSI'] = (data['RSI'] > 0.70).astype(float)
        data['Oversold_RSI'] = (data['RSI'] < 0.30).astype(float)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        if not data.empty:
            cleaned_dfs.append(data)

    return cleaned_dfs