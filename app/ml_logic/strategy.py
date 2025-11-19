import torch
import pandas as pd
from app.data.yfinance_fetcher import get_historical_data
from app.data.ticker_source import get_sp500_tickers
from app.data.preprocessor_utils import normalize_window, to_seq
from app.ml_logic.pred_models.only_close_model import pred_next_day_no_ticker, fine_tune_model, setup_pred_model
from app.utils import get_date_range
from app.ml_logic.model_loader import load_model_artifacts, save_model_artifacts
import math
from config import *


def optimal_picks(model_prefix, today = pd.Timestamp.today().normalize()):
    model_state_dict, config = load_model_artifacts(model_prefix)
    is_quantized = "quantized" in model_prefix
    model = setup_pred_model(model_state_dict, config, is_quantized)

    start, end = get_date_range("3M", today)
    sp_tickers = get_sp500_tickers()
    if not sp_tickers:
        return None, None
    sp_tickers.append("^SPX")

    processed_tickers = [t.replace(".", "-") for t in sp_tickers]
    all_historical_data = get_historical_data(processed_tickers, start, end)
    all_close_data = all_historical_data["Close"]

    all_predictions = []
    for ticker in processed_tickers:
        try:
            input_tensor, latest_close_price, mean, std = _prepare_data_for_prediction(all_close_data[ticker], ticker)
            prediction = pred_next_day_no_ticker(input_tensor, model, mean, std)
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
    data = get_historical_data(ticker, start, end)
    close_data = data["Close"]
    input_tensor, _, mean, std = _prepare_data_for_prediction(close_data, ticker)
    prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

    print(f"Predicted value: {prediction}")
    return prediction

def get_historical_volatility(ticker, start, end):
    df = get_historical_data(ticker, start, end)
    try:
        df['Returns'] = df['Close'].pct_change(fill_method=None).dropna()
        daily_variance = df['Returns'].var()
        annualized_variance = daily_variance * 252
        return annualized_variance
    except Exception as e:
        print(f"ticker = {ticker} df = {df} e = {e}")
        return 10000000


def _prepare_data_for_prediction(data, ticker):
    if len(data) < SEQUENCE_SIZE:
        raise ValueError(f"Insufficient data for {ticker}. Need at least {SEQUENCE_SIZE} points, got {len(data)}.")

    latest_close_price = data.iloc[-1]
    input_sequence = data[-SEQUENCE_SIZE:]

    normalized_window, mean, std = normalize_window(input_sequence)
    input_tensor = torch.tensor(normalized_window.to_numpy(), dtype=torch.float32).unsqueeze(0).unsqueeze(2)

    return input_tensor, latest_close_price, mean, std


def calculate_kelly_allocations(model, end=None):

    # if end is none, get range until today, end = today
    # if end is not none, backtest
    start, end = get_date_range(lookback_period, end)

    predictions, spy_delta = optimal_picks(model, end)
    if not predictions:
        print("No predictions available to calculate Kelly bets.")
        return None

    kelly_allocations = []
    for ticker, predicted_delta in predictions:
        mu = predicted_delta
        if math.isnan(mu):
            print(f"Skipping {ticker}: mu is NaN")
            continue
        try:
            sigma_squared = get_historical_volatility(ticker, start, end)
            if sigma_squared is None or math.isnan(sigma_squared) or sigma_squared <= 0:
                print(f"Skipping {ticker}: sigma^2 is NaN or vol neg")
                continue

            kelly_fraction = (mu - RISK_FREE_RATE) / sigma_squared
            allocation = kelly_fraction * HALF_KELLY_MULTIPLIER

            if allocation < 0:
                allocation = 0.0
            elif allocation > 1.0:
                allocation = 1.0

            if math.isnan(allocation) or math.isinf(allocation):
                print(f"Skipping {ticker}: allocation ended up NaN/inf")
                continue

            kelly_allocations.append((ticker, allocation, mu, sigma_squared))
        except Exception as e:

            print(f"Warning {model}: Skipping {ticker} from Kelly allocation due to data/calculation error: {e}")
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


def tune(model, date):
    model_dict, config = load_model_artifacts(model)
    start, end = get_date_range("3M", date)

    snp = get_sp500_tickers()
    if not snp:
        return False
    list_of_df = []
    for ticker in snp:
        list_of_df.append(get_historical_data(ticker, start, end))
    new_model_dict, new_config = fine_tune_model(model_dict, config, list_of_df)
    save_model_artifacts(model, new_model_dict, new_config)
    return True