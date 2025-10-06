import os
import argparse
import torch
from torch import nn, optim
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from data import DataHandler, to_seq
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

class StockTransformerModel(nn.Module):
    def __init__(self, num_features_in, embedding_dim, max_len=50, num_layers=3):
        super().__init__()
        self.feature_embedding = nn.Linear(in_features=num_features_in, out_features=embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            batch_first=True # batch_size (64), seq_len(50), d_model embedding_dim = 256
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.final_layers = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        feature_embeddings = self.feature_embedding(x)
        combined_embeddings = self.positional_encoding(feature_embeddings)
        output = self.transformer_encoder(combined_embeddings)
        prediction = self.final_layers(output[:, -1, :])
        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pe_slice = self.pe[:seq_len].transpose(0, 1)
        x = x + pe_slice
        return self.dropout(x)


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(os.path.join(model_dir, "model.pth"), map_location=device)
    config = checkpoint["config"]

    model = StockTransformerModel(
        num_features_in=config["num_features_in"],
        embedding_dim=config["embedding_dim"],
        max_len=config["max_len"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model.to(device)

# TODO: maybe add unequal weighting, like thesandp??????
def train_fn(args):
    print("CALLED NEW ATTEMPT TRAIN FN")

    # --- 1. Data Loading ---
    data_handler = DataHandler()
    list_of_dfs = data_handler.get_dfs_from_s3(prefix="historical_data/")

    list_train_df = []
    list_test_df = []
    for df in list_of_dfs:
        split_idx = int(len(df) * 0.8)
        list_train_df.append(df[:split_idx])
        list_test_df.append(df[split_idx:])

    list_close_train = []
    list_close_test = []

    for df in list_train_df:
        list_close_train.append(df["Close"].to_numpy().reshape(-1, 1))

    for df in list_test_df:
        list_close_test.append(df["Close"].to_numpy().reshape(-1, 1))

    #all_train = np.concatenate(list_close_train, axis=0)

    def normalize_window(window):
        mean = window.mean()
        std = window.std() + 1e-8
        return (window - mean) / std, mean, std

    # --- 5. Build Training Data ---
    SEQUENCE_SIZE = 50
    norm_windows, targets = [], []
    for arr in list_close_train:
        xt, yt = to_seq(SEQUENCE_SIZE, arr)
        for x_win, y_val in zip(xt, yt):
            x_norm, mean, std = normalize_window(x_win)
            norm_windows.append(x_norm)
            targets.append((y_val - mean) / std)

    x_train = np.stack(norm_windows)

    y_train = np.array(targets, dtype=np.float32).reshape(-1, 1)

    # --- 6. Build Testing Data (same normalization) ---
    norm_windows_test, targets_test = [], []
    for arr in list_close_test:
        xt, yt = to_seq(SEQUENCE_SIZE, arr)
        for x_win, y_val in zip(xt, yt):
            x_norm, mean, std = normalize_window(x_win)
            norm_windows_test.append(x_norm)
            targets_test.append((y_val - mean) / std)


    x_test = np.stack(norm_windows_test)

    y_test = np.array(targets_test, dtype=np.float32).reshape(-1, 1)

    # Convert to torch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    num_features = x_train.shape[2]
    embedding_dim = 256
    max_len = 50

    model = StockTransformerModel(num_features, embedding_dim, max_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)
    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    model.to(device)

    early_stop_count = 0
    min_val_loss = float("inf")

    best_checkpoint = None
    # --- 3. Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        has_printed = False
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)

            if not has_printed:
                # The input sequence (x_value) for the first sample (50 normalized features)
                x_sequence = x_batch[0].cpu().numpy().flatten()

                print(f"\n--- Epoch {epoch + 1} Sample 1 Status (First Batch) ---")
                # Print only a summary of the 50-element input sequence for brevity
                print(f"Input X Sequence (Normalized, first 5 elements): {x_sequence[:5].tolist()}")
                print(f"Input X Sequence (Normalized, last 5 elements): {x_sequence[-5:].tolist()}")
                print(f"True Y Value (Normalized): {y_batch[0].item():.6f}")
                print(f"Predicted Y Value (Normalized): {predictions[0].item():.6f}")
                print("-----------------------------------------------------")

                has_printed = True  # Ensure it only prints once per epoch

            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_count = 0
            best_checkpoint = {
                "model_state": model.state_dict(),
                "config": {
                    "num_features_in": num_features,
                    "embedding_dim": embedding_dim,
                    "max_len": max_len,
                },
            }
        else:
            early_stop_count += 1
        if early_stop_count > 5:
            print("Early stopping")
            break

        print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {val_loss:.4f}")

    #scaler_path = os.path.join(args.model_dir, "scaler.joblib")
    #joblib.dump(scaler, scaler_path)
    #print(f"scaler to {scaler_path}")

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(best_checkpoint, path)
    print(f"Model saved to {path}")


# =====================================================
#  Main Entry (for SageMaker Training Job)
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--use-cuda", type=bool, default=False)

    # Directories
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])

    args = parser.parse_args()

    train_fn(args)