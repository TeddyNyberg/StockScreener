import torch
import pandas as pd
from app.data.yfinance_fetcher import get_historical_data
from app.data.ticker_source import get_sp500_tickers
from app.data.preprocessor_utils import normalize_window
from app.ml_logic.pred_models.only_close_model import pred_next_day_no_ticker, fine_tune_model_daily
from app.utils import get_date_range
from app.ml_logic.model_loader import load_model_artifacts
from config import *


def optimal_picks(today = pd.Timestamp.today().normalize()):
    model_state_dict, config = load_model_artifacts()

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

    return all_predictions, all_predictions[-1]

def predict_single_ticker(ticker):
    model_state_dict, config = load_model_artifacts()

    start, end = get_date_range("3M")
    input_tensor, _, mean, std = _prepare_data_for_prediction(ticker, start, end)
    prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

    print(f"Predicted value: {prediction}")
    return prediction

def get_historical_volatility(ticker, start, end):
    df = get_historical_data(ticker, start, end)
    df['Returns'] = df['Close'].pct_change().dropna()
    daily_variance = df['Returns'].var()
    annualized_variance = daily_variance * 252
    return annualized_variance


def _prepare_data_for_prediction(ticker, start, end):
    df = get_historical_data(ticker, start, end)
    data = df["Close"]

    if len(data) < SEQUENCE_SIZE:
        raise ValueError(f"Insufficient data for {ticker}. Need at least {SEQUENCE_SIZE} points, got {len(data)}.")

    latest_close_price = df["Close"].iloc[-1]
    input_sequence = data[-SEQUENCE_SIZE:]

    normalized_window, mean, std = normalize_window(input_sequence)
    input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0)

    return input_tensor, latest_close_price, mean, std


def calculate_kelly_allocations(end=None):

    # if end is none, get range until today, end = today
    # if end is not none, backtest
    start, end = get_date_range(lookback_period, end)

    predictions, spy_delta = optimal_picks(end)
    if not predictions:
        print("No predictions available to calculate Kelly bets.")
        return None

    kelly_allocations = []
    for ticker, predicted_delta in predictions:
        mu = predicted_delta
        try:
            sigma_squared = get_historical_volatility(ticker, start, end)

            # Ensure variance is positive to avoid division by zero or errors
            if sigma_squared <= 0:
                print(f"Skipping {ticker}: Volatility (sigma^2) is zero or negative.")
                continue

            kelly_fraction = (mu - RISK_FREE_RATE) / sigma_squared
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

def load_for_tuning():
    model_state_dict, config = load_model_artifacts()
    new_data = None
    fine_tune_model_daily(model_state_dict, config, new_data)
    pass


