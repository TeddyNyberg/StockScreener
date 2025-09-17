import io
import torch
from torch.utils.data import Dataset
import pandas as pd
import boto3
import yfinance as yf
import requests
import numpy as np

S3_BUCKET_NAME = "stock-screener-bucker"



# srtore this result? so i dont need ot make it new everytime? idk whats more efficient??????
# dfs already have high, low, close, open, volume
# add features
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

        data["Average_Move"] = data["Delta"].rolling(window=20).mean()

        data.dropna(inplace=True)

    return list_of_df


def df_to_tensor_with_dynamic_ids(list_of_df, ticker_map, window_size=50):
    tensor_list = []

    for df in list_of_df:
        ticker_symbol = df["Ticker"].iloc[0]

        ticker_id = ticker_map.get(ticker_symbol, -1)
        if ticker_id == -1:
            print(f"Warning: Ticker '{ticker_symbol}' not found in the trained model's map.")
            continue  # Skip this ticker as the model doesn't know it.

        # Make a copy and drop non-numeric columns
        df_copy = df.copy()
        df_copy['ticker_id'] = ticker_id
        df_copy = df_copy.drop(columns=['Ticker'])
        df_copy = df_copy.select_dtypes(include=[np.number])

        values = df_copy.to_numpy(dtype='float32')

        # Generate sliding windows
        for i in range(len(values) - window_size + 1):
            window = values[i: i + window_size]
            tensor_list.append(torch.from_numpy(window))

    return tensor_list







# to use
# stock_dataset = StockDataset(data_tensors)
# dataloader = DataLoader(stock_dataset, batch_size=500, shuffle=True)
# num_feat = 23 rn
# embedding dim = 256 or 512 or 718
# seq_len = days of hist to use = 50
# linear_projection = nn.Linear(in_features=num_features, out_features=embedding_dim)
# embedded_data = linear_projection(what dataloader gives)
class StockDataset(Dataset): # make iterable instead?
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    stock_data['Ticker'] = ticker
    return stock_data


def get_sp500_tickers():
    url = "https://stockanalysis.com/list/sp-500-stocks/"
    r = requests.get(url)
    print(r.status_code)
    try:
        tables = pd.read_html(io.StringIO(r.text))
        print(tables)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []


class DataHandler:
    def __init__(self):
        self.s3_client = boto3.client('s3')

    def save_to_s3(self, df, file_path):
        if df is None or df.empty:
            print("DataFrame is empty, skipping S3 upload.")
            return

        print(f"Saving data to S3 at s3://{S3_BUCKET_NAME}/{file_path}...")
        try:

            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, engine='pyarrow')

            self.s3_client.put_object(
                Bucket=S3_BUCKET_NAME,
                Key=file_path,
                Body=parquet_buffer.getvalue()
            )
            print("Successfully uploaded to S3.")

        except Exception as e:
            print(f"Error uploading to S3: {e}")

    def get_dfs_from_s3(self, prefix=''):

        if self.s3_client is None:
            s3_client = boto3.client('s3')

        list_of_dfs = []

        try:
            response = self.s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=prefix)

            if 'Contents' not in response:
                print(f"No objects found in bucket '{S3_BUCKET_NAME}' with prefix '{prefix}'.")
                return list_of_dfs

            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith('.parquet'):
                    print(f"Reading file: {key}")

                    file_object = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=key)

                    df = pd.read_parquet(io.BytesIO(file_object['Body'].read()))
                    list_of_dfs.append(df)

        except Exception as e:
            print(f"Error retrieving data from S3: {e}")
            return []

        return list_of_dfs



