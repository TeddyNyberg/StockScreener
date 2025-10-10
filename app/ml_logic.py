import boto3
import io
import tarfile
import torch
from data import fetch_stock_data, normalize_window, get_sp500_tickers
from model import pred_next_day_no_ticker
from search import get_date_range
from settings import *

s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACC_KEY_ID,
        aws_secret_access_key= AWS_SCR_ACC_KEY
    )


def optimal_picks():
    print("Fetching model artifacts from S3...")

    model_buffer = io.BytesIO()

    s3_client.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARTIFACTS_PREFIX, model_buffer)
    model_buffer.seek(0)
    print("Downloaded entire model archive to memory.")

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        # Load the PyTorch model state dict
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))
            print("Model state dict loaded successfully.")

    model_state_dict = checkpoint.get("model_state")
    config = checkpoint.get("config")
    if model_state_dict is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    start, end = get_date_range("6M")
    sp_tickers = get_sp500_tickers()
    sp_tickers.append("^SPX")

    all_predictions = []
    for ticker in sp_tickers:
        if "." in ticker:
            ticker = ticker.replace(".", "-")
        df = fetch_stock_data(ticker, start, end)
        data = df["Close"]
        seq_size = 50
        input_sequence = data[-seq_size:]

        normalized_window, mean, std = normalize_window(input_sequence)

        input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0)
        prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

        delta = ((prediction - df["Close"].iloc[-1]) / df["Close"].iloc[-1]).item()

        all_predictions.append((ticker, delta))

    sorted_predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)

    print("\n--- Sorted Predictions (Highest Predicted Return First) ---")

    for ticker, delta in sorted_predictions[:10]:
        print(f"Stock: {ticker}, Predicted Delta: {delta:+.4f}")

    print("\n--- Lowest Predictions ---")
    for ticker, delta in sorted_predictions[-5:]:
        print(f"Stock: {ticker}, Predicted Delta: {delta:+.4f}")

    if sorted_predictions:
        best_stock, best_value = sorted_predictions[0]
        print(f"\nOverall Best Stock Pick: {best_stock}")
        print(f"Predicted Value: {best_value:+.4f}")
    else:
        print("No predictions were successfully generated.")

    return sorted_predictions, all_predictions[-1]


def predict_single_ticker(ticker):
    print(f"Fetching model artifacts for {ticker} from S3...")

    model_buffer = io.BytesIO()

    s3_client.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARTIFACTS_PREFIX, model_buffer)
    model_buffer.seek(0)

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))

    model_state_dict = checkpoint.get("model_state")
    config = checkpoint.get("config")
    if model_state_dict is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    start, end = get_date_range("6M")
    df = fetch_stock_data(ticker, start, end)

    data = df["Close"]
    seq_size = 50
    input_sequence = data[-seq_size:]

    normalized_window, mean, std = normalize_window(input_sequence)

    input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0)
    prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

    print(f"Predicted value: {prediction}")
    return prediction

def get_historical_volatility(ticker, start, end):
    df = fetch_stock_data(ticker, start, end)
    df['Returns'] = df['Close'].pct_change().dropna()
    daily_variance = df['Returns'].var()
    annualized_variance = daily_variance * 252
    return annualized_variance