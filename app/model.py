import os, io
import argparse
import torch
from torch import nn as nn
from torch import optim as optim
import math
from data import feat_engr, df_to_tensor_with_dynamic_ids, StockDataset, DataHandler
from torch.utils.data import DataLoader
import joblib
import numpy as np
from newattemptmodel import StockTransformerModel


batch_size = 64
sequence_length = 0
num_features = 0


class StockTransformerModelNull(nn.Module):

    def __init__(self, num_features_in, embedding_dim, num_tickers, max_len=1000):
        print("MAKING MODEL INIT")
        super().__init__()
        sequence_length = max_len
        num_features = num_features_in
        self.feature_embedding = nn.Linear(
            in_features=num_features - 1,  # Subtract 1 because we're taking the ticker ID out
            out_features=embedding_dim
        )

        self.ticker_embedding = nn.Embedding(
            num_embeddings=num_tickers,
            embedding_dim=embedding_dim
        )

        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=embedding_dim * 4
        )

        self.final_linear = nn.Linear(embedding_dim, 1)

    def forward(self, x):

        feature_data = x[:, :, :-1]
        ticker_ids = x[:, 0, -1].long()

        feature_embeddings = self.feature_embedding(feature_data)

        ticker_embeddings = self.ticker_embedding(ticker_ids).unsqueeze(1)
        ticker_embeddings = ticker_embeddings.expand_as(feature_embeddings)

        combined_embeddings = feature_embeddings + ticker_embeddings

        combined_embeddings = self.positional_encoding(combined_embeddings)

        output = self.transformer_encoder_layer(combined_embeddings)

        prediction = self.final_linear(output[:, -1, :])

        return prediction


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        print("POS ENCODING INIT")
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is the embedded input tensor with shape (batch_size, sequence_length, embedding_dim)
        return x + self.pe[:, :x.size(1)]



def train_model():
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, _ = parser.parse_known_args()

    print("CALLED TRAIN MODEL")
    data_handler = DataHandler()

    list_of_dfs = data_handler.get_dfs_from_s3(prefix='historical_data/')

    processed_dfs = feat_engr(list_of_dfs)

    split_idx = int(len(processed_dfs) * 0.8)
    training_dfs = processed_dfs[:split_idx]
    test_dfs = processed_dfs[split_idx:]

    global TICKER_TO_ID_MAP, map_ind

    TICKER_TO_ID_MAP = {}
    map_ind = 0

    for df in processed_dfs:
        if df["Ticker"].iloc[0] not in TICKER_TO_ID_MAP.keys():
            TICKER_TO_ID_MAP[df["Ticker"].iloc[0]] = map_ind
            map_ind += 1

    training_tensors = df_to_tensor_with_dynamic_ids(training_dfs, TICKER_TO_ID_MAP)
    test_tensors = df_to_tensor_with_dynamic_ids(test_dfs, TICKER_TO_ID_MAP)

    train_dataset = StockDataset(training_tensors)
    test_dataset = StockDataset(test_tensors)

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data

    # --- 4. Model, Loss, and Optimizer Initialization ---
    num_features = training_tensors[0].shape[1]
    # TICKER_TO_ID_MAP is assumed to be a global variable
    num_tickers = len(TICKER_TO_ID_MAP)
    embedding_dim = 256
    max_len = 1000
    num_epochs = args.epochs

    model = StockTransformerModelNull(num_features, embedding_dim, num_tickers, max_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("Starting training...")

    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        print(train_dataloader)
        for batch_data in train_dataloader:
            inputs = batch_data[:, :-1, :]
            targets = batch_data[:, -1, 0].unsqueeze(1)

            predictions = model(inputs)

            loss = criterion(predictions, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            print("Checking gradients for feature embedding...")
            for name, param in model.named_parameters():
                if 'feature_embedding' in name and param.grad is not None:
                    print(f"Gradient norm for {name}: {param.grad.norm()}")

            print("Checking gradients for ticker embedding...")
            for name, param in model.named_parameters():
                if 'ticker_embedding' in name and param.grad is not None:
                    print(f"Gradient norm for {name}: {param.grad.norm()}")
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {loss.item():.4f}")

    # --- 6. Evaluation Loop ---
    print("\nStarting evaluation...")
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for batch_data in test_dataloader:
            inputs = batch_data[:, :-1, :]
            targets = batch_data[:, -1, 0].unsqueeze(1)

            predictions = model(inputs)

            test_loss += criterion(predictions, targets).item()

    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

    with open(os.path.join(args.model_dir, 'ticker_to_id.pt'), 'wb') as f:
        torch.save(TICKER_TO_ID_MAP, f)


    print("Finished training.")


def pred_next_day(input_tensor, ticker_to_id_map, model_state_dict, scaler):

    embedding_dim = 256
    num_tickers = len(ticker_to_id_map)
    max_len = 50

    # TODO: change this so it takes in model is an arg then you can pred next day w any model
    # also this would allow for when i call it 500 times, i dont need to load it 500 times.
    model = StockTransformerModel(input_tensor.shape[-1], embedding_dim, num_tickers, max_len)
    model.load_state_dict(model_state_dict)
    model.eval()

    with torch.no_grad():

        normalized_prediction = model(input_tensor)

    close_idx = 0  # Assuming 'Close' is the first feature
    close_min = scaler.data_min_[close_idx]
    close_max = scaler.data_max_[close_idx]

    prediction = normalized_prediction.item() * (close_max - close_min) + close_min

    return prediction.item()


def pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std):
    model = StockTransformerModel(
        num_features_in=config["num_features_in"],
        embedding_dim=config["embedding_dim"],
        max_len=config["max_len"]
    )

    # TODO: change this so it takes in model is an arg then you can pred next day w any model
    #      also this would allow for when i call it 500 times, i dont need to load it 500 times.
    model.load_state_dict(model_state_dict)
    model.eval()

    with torch.no_grad():
        normalized_prediction = model(input_tensor)
    last_prediction = normalized_prediction[-1][0]
    prediction_np = np.array(last_prediction.item()).reshape(1, -1)
    prediction = (prediction_np[0][0] * std) + mean

    return prediction.item()
