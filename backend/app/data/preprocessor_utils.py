# normalization, seq_create
import torch
from torch.utils.data import Dataset
import numpy as np
from backend.app.config import *


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


class TranscriptDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = str(row['transcript_masked'])

        label = float(row['return_percentile'])

        ticker = row['ticker']
        date = str(row['date']).split()[0]
        doc_id = f"{ticker}_{date}"

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_overflowing_tokens=True,
            stride=45
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {
            'input_ids': input_ids,  # Shape: [num_chunks, 512]
            'attention_mask': attention_mask,  # Shape: [num_chunks, 512]
            'label': torch.tensor(label, dtype=torch.float),
            'id': doc_id,
            'raw_text': text[:100] + "..."
        }


def prepare_data_for_prediction(data):

    latest_close_price = data.iloc[-1]
    input_sequence = data[-SEQUENCE_SIZE:]

    normalized_window, mean, std = normalize_window(input_sequence)

    if hasattr(mean, 'item'):
        mean = mean.item()
    if hasattr(std, 'item'):
        std = std.item()

    input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0)

    return input_tensor, latest_close_price, mean, std


def to_multi_seq(seq_size, data_array, target_col_idx):
    x = []
    y = []

    for i in range(len(data_array) - seq_size):
        window = data_array[i: i + seq_size]
        target = data_array[i + seq_size, target_col_idx]
        x.append(window)
        y.append(target)

    return np.array(x), np.array(y)


def normalize_multivariate_window(window, target_val, target_col_idx, indices_to_norm):

    selected_cols = window[:, indices_to_norm]
    means = np.mean(selected_cols, axis=0)
    stds = np.std(selected_cols, axis=0)
    stds[stds == 0] = 1.0

    norm_window = window.copy()

    norm_window[:, indices_to_norm] = (selected_cols - means) / stds

    target_mean = np.mean(window[:, target_col_idx])
    target_std = np.std(window[:, target_col_idx])
    if target_std == 0: target_std = 1.0

    norm_target = (target_val - target_mean) / target_std

    return norm_window, norm_target, target_mean, target_std