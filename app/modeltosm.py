import os
import argparse
import torch
from torch import nn, optim
import math
from torch.utils.data import DataLoader
from data import feat_engr, df_to_tensor_with_dynamic_ids, StockDataset, DataHandler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

class StockTransformerModel(nn.Module):
    def __init__(self, num_features_in, embedding_dim, num_tickers, max_len=1000):
        super().__init__()
        print("MAKING MODEL INIT")
        self.feature_embedding = nn.Linear(
            in_features=num_features_in - 1,  # Subtract 1 because ticker ID is separate
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
        super().__init__()
        print("POS ENCODING INIT")

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(os.path.join(model_dir, "model.pth"), map_location=device)
    config = checkpoint["config"]

    model = StockTransformerModel(
        num_features_in=config["num_features_in"],
        embedding_dim=config["embedding_dim"],
        num_tickers=config["num_tickers"],
        max_len=config["max_len"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model.to(device)


feature_columns = ['Close', 'Volume', 'Open', 'High', "Low", "Range", "Delta", "Delta_Percent",
                       "Vol_vs_Avg", "Large_Move", "Large_Up", "Large_Down", "Trend_Up", "Trend_Down",
                       "Break_Up", "Break_Down", "BB_Upper", "BB_Lower", "Cross_BB_Upper",
                       "Cross_BB_Lower", "RSI", "Overbought_RSI", "Oversold_RSI", "Average_Move"]

def get_scaler(list_of_dfs):
    scaler = MinMaxScaler()
    processed_dfs = feat_engr(list_of_dfs)
    split_idx = int(len(processed_dfs) * 0.8)
    train_dfs = processed_dfs[:split_idx]
    all_train_dfs = pd.concat(train_dfs)
    scaler.fit(all_train_dfs[feature_columns])
    return scaler


def train_fn(args):
    """Custom training loop for SageMaker training container."""

    print("CALLED TRAIN FN")

    # --- 1. Data Loading ---
    data_handler = DataHandler()
    list_of_dfs = data_handler.get_dfs_from_s3(prefix="historical_data/")
    processed_dfs = feat_engr(list_of_dfs)

    split_idx = int(len(processed_dfs) * 0.8)
    training_dfs = processed_dfs[:split_idx]
    test_dfs = processed_dfs[split_idx:]

    scaler = get_scaler(list_of_dfs)

    # Transform the data using the same feature_columns list
    def transform_df(df):
        df[feature_columns] = scaler.transform(df[feature_columns])
        return df

    training_dfs = [transform_df(df) for df in training_dfs]
    test_dfs = [transform_df(df) for df in test_dfs]

    # Map tickers to IDs
    ticker_to_id_map = {}
    for df in processed_dfs:
        ticker = df["Ticker"].iloc[0]
        if ticker not in ticker_to_id_map:
            ticker_to_id_map[ticker] = len(ticker_to_id_map)

    training_tensors = df_to_tensor_with_dynamic_ids(training_dfs, ticker_to_id_map)
    test_tensors = df_to_tensor_with_dynamic_ids(test_dfs, ticker_to_id_map)

    train_dataset = StockDataset(training_tensors)
    test_dataset = StockDataset(test_tensors)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # --- 2. Model Setup ---
    num_features = training_tensors[0].shape[1]
    num_tickers = len(ticker_to_id_map)
    embedding_dim = 256
    max_len = 1000

    model = StockTransformerModel(num_features, embedding_dim, num_tickers, max_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    model.to(device)

    # --- 3. Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            inputs = batch_data[:, :-1, :]
            targets = batch_data[:, -1, 0].unsqueeze(1)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}")

    # --- 4. Evaluation ---
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            inputs = batch_data[:, :-1, :]
            targets = batch_data[:, -1, 0].unsqueeze(1)

            predictions = model(inputs)
            test_loss += criterion(predictions, targets).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Average Test Loss: {avg_test_loss:.4f}")

    # --- 5. Save checkpoint (weights + config) ---
    ticker_map_path = os.path.join(args.model_dir, "ticker_to_id.pt")
    torch.save(ticker_to_id_map, ticker_map_path)
    print(f"Ticker map saved to {ticker_map_path}")

    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "num_features_in": num_features,
            "embedding_dim": embedding_dim,
            "num_tickers": num_tickers,
            "max_len": max_len,
        },
    }

    print(scaler)
    print(scaler.data_min_)
    scaler_path = os.path.join(args.model_dir, "scaler.pt")
    torch.save(scaler, scaler_path)
    print(f"scaler to {scaler_path}")

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")



# =====================================================
#  Main Entry (for SageMaker Training Job)
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--use-cuda", type=bool, default=False)

    # Directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])

    args = parser.parse_args()

    train_fn(args)