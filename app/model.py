
import torch
from torch import nn as nn
from torch import optim as optim
import torch.utils.data as data
import math
from data import feat_engr, df_to_tensor_with_dynamic_ids, StockDataset
from torch.utils.data import DataLoader
import copy

batch_size = 64
sequence_length = 0
num_features = 0
class StockTransformerModel(nn.Module):
    def __init__(self, num_features_in, embedding_dim, num_tickers, max_len=1000):
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


def train_model(list_of_dfs):

    processed_dfs = feat_engr(list_of_dfs)

    split_idx = int(len(processed_dfs) * 0.8)
    training_dfs = processed_dfs[:split_idx]
    test_dfs = processed_dfs[split_idx:]

    global TICKER_TO_ID_MAP, MAP_IND

    TICKER_TO_ID_MAP = {}
    MAP_IND = 0

    for df in processed_dfs:
        if df["Ticker"].iloc[0] not in TICKER_TO_ID_MAP.keys():
            TICKER_TO_ID_MAP[df["Ticker"].iloc[0]] = MAP_IND
            MAP_IND += 1

    torch.save(TICKER_TO_ID_MAP, "ticker_to_id.pt")

    training_tensors = df_to_tensor_with_dynamic_ids(training_dfs, TICKER_TO_ID_MAP)
    test_tensors = df_to_tensor_with_dynamic_ids(test_dfs, TICKER_TO_ID_MAP)


    """
    train_dataset = StockDataset(training_tensors)
    test_dataset = StockDataset(test_tensors)
    
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data

    # --- 4. Model, Loss, and Optimizer Initialization ---
    num_features = training_tensors[0].shape[1]
    # TICKER_TO_ID_MAP is assumed to be a global variable
    num_tickers = len(TICKER_TO_ID_MAP)
    embedding_dim = 256
    max_len = 1000
    num_epochs = 10

    model = StockTransformerModel(num_features, embedding_dim, num_tickers, max_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


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

    torch.save(model.state_dict(), "stock_model.pt")
    print("Finished")"""


def pred_next_day(tensor, TICKER_TO_ID_MAP):
    embedding_dim = 256
    num_tickers = len(TICKER_TO_ID_MAP)
    max_len = 1000
    print("pred_next_day in model.py")
    print(TICKER_TO_ID_MAP)
    print(num_tickers)
    print(tensor.shape)
    model = StockTransformerModel(tensor.shape[-1], embedding_dim, num_tickers, max_len)
    model.load_state_dict(torch.load("stock_model.pt"))
    model.eval()
    print(model(tensor))
