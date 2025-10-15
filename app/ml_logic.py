import boto3
import io
import tarfile
import torch
import pandas as pd
from data import fetch_stock_data, normalize_window, get_sp500_tickers
from model import pred_next_day_no_ticker
from search import get_date_range
from settings import *

s3_client = boto3.client(
        's3',
        aws_access_key_id = AWS_ACC_KEY_ID,
        aws_secret_access_key= AWS_SCR_ACC_KEY
    )

_MODEL_STATE_DICT = None
_CONFIG = None

def optimal_picks(today = pd.Timestamp.today().normalize()):

    model_state_dict, config = _load_model_artifacts()

    start, end = get_date_range("3M", today)
    sp_tickers = get_sp500_tickers()
    sp_tickers.append("^SPX")

    all_predictions = []
    for ticker in sp_tickers:
        if "." in ticker:
            ticker = ticker.replace(".", "-")

        try:
            input_tensor, latest_close_price, mean, std = _prepare_data_for_prediction(ticker, start, end)
            prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)
            delta = ((prediction - latest_close_price) / latest_close_price).item()
            all_predictions.append((ticker, delta))
        except ValueError as e:
            print(f"Warning: Skipping {ticker}. {e}")
            continue
        except Exception as e:
            print(f"Warning: Skipping {ticker} for prediction due to data/network error: {e}")
            continue

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

    model_state_dict, config = _load_model_artifacts()

    start, end = get_date_range("3M")
    input_tensor, _, mean, std = _prepare_data_for_prediction(ticker, start, end)
    prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

    print(f"Predicted value: {prediction}")
    return prediction

def get_historical_volatility(ticker, start, end):
    df = fetch_stock_data(ticker, start, end)
    df['Returns'] = df['Close'].pct_change().dropna()
    daily_variance = df['Returns'].var()
    annualized_variance = daily_variance * 252
    return annualized_variance


def _prepare_data_for_prediction(ticker, start, end):
    df = fetch_stock_data(ticker, start, end)
    data = df["Close"]

    seq_size = 50
    if len(data) < seq_size:
        raise ValueError(f"Insufficient data for {ticker}. Need at least {seq_size} points, got {len(data)}.")

    latest_close_price = df["Close"].iloc[-1]
    input_sequence = data[-seq_size:]

    normalized_window, mean, std = normalize_window(input_sequence)

    input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0)

    return input_tensor, latest_close_price, mean, std


def _load_model_artifacts():
    global _MODEL_STATE_DICT, _CONFIG

    if _MODEL_STATE_DICT is not None and _CONFIG is not None:
        print("Using cached model artifacts.")
        return _MODEL_STATE_DICT, _CONFIG

    print("Loading model artifacts from S3...")

    model_buffer = io.BytesIO()

    s3_client.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARTIFACTS_PREFIX, model_buffer)
    model_buffer.seek(0)
    print("Downloaded entire model archive to memory.")

    with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:
        with tar.extractfile('model.pth') as f:
            checkpoint = torch.load(io.BytesIO(f.read()))

    _MODEL_STATE_DICT = checkpoint.get("model_state")
    _CONFIG = checkpoint.get("config")

    if _MODEL_STATE_DICT is None:
        raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

    print("Model state dict and config loaded successfully.")
    return _MODEL_STATE_DICT, _CONFIG


def calculate_kelly_allocations(lookback_period="6M", end=None):

    if end is None:
        start, end = get_date_range(lookback_period)
    else:
        start, end = get_date_range(lookback_period, end)

    sorted_predictions, spy_delta = optimal_picks(end)

    if not sorted_predictions:
        print("No predictions available to calculate Kelly bets.")
        return

    RISK_FREE_RATE = 0.005  # Assuming 0.5% annualized risk-free rate (r)

    kelly_allocations = []

    for ticker, predicted_delta in sorted_predictions:
        mu = predicted_delta

        try:
            sigma_squared = get_historical_volatility(ticker, start, end)

            # Ensure variance is positive to avoid division by zero or errors
            if sigma_squared <= 0:
                print(f"Skipping {ticker}: Volatility (sigma^2) is zero or negative.")
                continue

            kelly_fraction = (mu - RISK_FREE_RATE) / sigma_squared

            # WE GOING FULL KELLY
            HALF_KELLY_MULTIPLIER = 1
            allocation = kelly_fraction * HALF_KELLY_MULTIPLIER

            if allocation < 0:
                allocation = 0.0
            elif allocation > 1.0:
                allocation = 1.0

            kelly_allocations.append((ticker, allocation, mu, sigma_squared))
        except Exception as e:
            # Handle any failure during volatility calculation and skip the ticker for the day.
            print(f"Warning: Skipping {ticker} from Kelly allocation due to data/calculation error: {e}")
            continue

    total_allocation = sum(allocation for _, allocation, _, _ in kelly_allocations)

    normalization_factor = 1.0 / total_allocation if total_allocation > 1.0 else 1.0

    print("\n--- Continuous Kelly-Based Position Sizing ---")
    print(f"Total Unnormalized Allocation: {total_allocation * 100:.2f}%")
    print(f"Normalization Factor (if > 100%): {normalization_factor:.4f}")

    final_allocations_unsorted = []
    for ticker, allocation, mu, sigma_squared in kelly_allocations:
        normalized_allocation = allocation * normalization_factor
        if normalized_allocation > 0:
            final_allocations_unsorted.append((ticker, normalized_allocation, mu))
            print(
                f"Stock: {ticker}, μ: {mu:+.4f}, σ²: {sigma_squared:.4f}, Allocation: {normalized_allocation * 100:.2f}%")

    final_allocations = sorted(final_allocations_unsorted, key=lambda x: x[1], reverse=True)

    return final_allocations