# normalization, seq_create
import torch
from torch.utils.data import Dataset
import numpy as np



def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, 0])
    return np.array(x), np.array(y)

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
        values[:, -1] = values[:, -1].astype('int64')

        # Generate sliding windows
        for i in range(len(values) - window_size + 1):
            window = values[i: i + window_size]
            tensor_list.append(torch.from_numpy(window))

    return tensor_list


def to_sequences(seq_size, feature_data, target_data):

    x = []
    y = []
    for i in range(len(feature_data) - seq_size):
        x.append(feature_data[i:i + seq_size])
        y.append(target_data[i + seq_size])

    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def to_seq(seq_size, obs):
    x = []
    y = []
    for i in range(len(obs) - seq_size):
        window = obs[i: (i+seq_size)]
        after_window = obs[i+ seq_size]
        x.append(window)
        y.append(after_window)
    return torch.tensor(x, dtype=torch.float32).view(-1, seq_size, 1), torch.tensor(y, dtype=torch.float32).view(-1,1)

def normalize_window(window):
    mean = window.mean()
    std = window.std() + 1e-8
    return (window - mean) / std, mean, std

class StockDataset(Dataset): # make iterable instead?
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]