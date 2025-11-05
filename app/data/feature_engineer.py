#feat_engr
import pandas as pd
import numpy as np

def feat_engr(list_of_df):
    for data in list_of_df:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data['Range'] = (data['High'] - data['Low']) / data['Open']
        data['Delta'] = data['Close'].diff()
        data['Delta_Percent'] = data['Close'].pct_change()
        data['Vol_vs_Avg'] = data['Volume'] / data['Volume'].rolling(window=20).mean().fillna(data["Volume"])
        data['Large_Move'] = (data['Delta_Percent'] >= 0.05) | (data['Delta_Percent'] <= -0.05).astype(int)
        data['Large_Down'] = (data['Delta_Percent'] <= -0.05).astype(int)
        data['Large_Up'] = (data['Delta_Percent'] >= 0.05).astype(int)
        data['Trend_Up'] = data['Delta'].rolling(window=5).apply(lambda x: (x > 0).all(), raw=True).fillna(0).astype(int)
        data['Trend_Down'] = data['Delta'].rolling(window=5).apply(lambda x: (x < 0).all(), raw=True).fillna(0).astype(int)
        data['Break_Up'] = np.where(data["Close"] > data["Close"].rolling(window=50).mean(), 1, 0)
        data['Break_Down'] = np.where(data["Close"] < data["Close"].rolling(window=50).mean(), 1, 0)

        data['BB_Upper'] = data['Close'].rolling(window=20).mean() + (data['Close'].rolling(window=20).std() * 2).fillna(np.inf)
        data['BB_Lower'] = data['Close'].rolling(window=20).mean() - (data['Close'].rolling(window=20).std() * 2).fillna(0)
        data['Cross_BB_Upper'] = np.where(data["Close"] > data['BB_Upper'], 1, 0)
        data['Cross_BB_Lower'] = np.where(data["Close"] < data['BB_Lower'], 1, 0)

        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs)).fillna(50)
        data['Overbought_RSI'] = (data['RSI'] > 70).astype(int)
        data['Oversold_RSI'] = (data['RSI'] < 30).astype(int)

        data["Average_Move"] = data["Delta_Percent"].rolling(window=20).mean()

        data.dropna(inplace=True)

    return list_of_df