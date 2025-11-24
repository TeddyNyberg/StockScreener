import torch
import pandas as pd
from app.data.yfinance_fetcher import get_historical_data
from app.data.ticker_source import get_sp500_tickers
from app.data.preprocessor_utils import normalize_window, to_seq, prepare_data_for_prediction
from app.ml_logic.pred_models.only_close_model import pred_next_day_no_ticker, fine_tune_model
from app.utils import get_date_range
from app.ml_logic.model_loader import load_model_artifacts, save_model_artifacts, load_model_artifacts_local
import time

from app.ml_logic.pred_models.only_close_model import pred_next_day_no_ticker, fine_tune_model, setup_pred_model
import math
import numpy as np
from config import *


def optimal_picks(model, today = pd.Timestamp.today().normalize()):
    model_state_dict, config = load_model_artifacts(model)

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
    try:
        df['Returns'] = df['Close'].pct_change(fill_method=None).dropna()
        daily_variance = df['Returns'].var()
        annualized_variance = daily_variance * 252
        return annualized_variance
    except Exception as e:
        print(f"ticker = {ticker} df = {df} e = {e}")
        return 10000000


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


def calculate_kelly_allocations(model, end=None):
    begin_time = time.perf_counter()

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
            if sigma_squared <= 0:
                print(f"Skipping {ticker}: Volatility (sigma^2) is zero or negative.")
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

    stop_time = time.perf_counter()
    print("time in one run og kelly: ", stop_time - begin_time)

    return final_allocations


def tune(model, date):
    model_dict, config = load_model_artifacts(model)
    start, end = get_date_range("3M", date)
    snp = get_sp500_tickers()
    list_of_df = []
    for ticker in snp:
        list_of_df.append(get_historical_data(ticker, start, end))
    new_model_dict, new_config = fine_tune_model(model_dict, config, list_of_df)
    save_model_artifacts(new_model_dict, new_config, model)




def get_all_volatilities(all_data):
    returns_df = all_data.pct_change(fill_method=None)
    returns_df = returns_df.dropna(how='all', axis=0)
    returns_df = returns_df.dropna(how='all', axis=1)

    daily_variance = returns_df.var()
    annualized_variance_series = daily_variance * 252 #252 trading days

    return annualized_variance_series



def calculate_kelly_allocations_new(model_version, is_quantized, end=None):

    begin_time = time.perf_counter()

    mus, all_vol_data = optimal_picks_new(model_version, is_quantized, end)
    all_closes = all_vol_data.iloc[-1]

    if mus is None or mus.empty:
        print("No predictions available to calculate Kelly bets.")
        return None
    volatility_series = get_all_volatilities(all_vol_data)


    sigma_squareds = volatility_series.reindex(mus.index, fill_value=-1)

    valid_mask = (sigma_squareds > 0) & (~np.isnan(sigma_squareds)) & (~np.isnan(mus))
    valid_mus = mus[valid_mask]
    valid_sigma_squareds = sigma_squareds[valid_mask]

    kelly_fraction = (valid_mus - RISK_FREE_RATE) / valid_sigma_squareds
    allocation = kelly_fraction * HALF_KELLY_MULTIPLIER
    kelly_allocations_series = np.clip(allocation, 0.0, 1.0)

    final_allocations_series = kelly_allocations_series[kelly_allocations_series > 0.0]

    total_allocation = final_allocations_series.sum()
    normalization_factor = 1.0 / total_allocation if total_allocation > 1.0 else 1.0

    print("\n--- Continuous Kelly-Based Position Sizing ---")
    print(f"Total Unnormalized Allocation: {total_allocation * 100:.2f}%")
    print(f"Normalization Factor (if > 100%): {normalization_factor:.4f}")

    normalized_allocations = final_allocations_series * normalization_factor

    final_allocations = []

    for ticker, normalized_allocation in normalized_allocations.items():
        mu = mus.loc[ticker]
        final_allocations.append((ticker, normalized_allocation, mu))
        print(f"Stock: {ticker}, μ: {mu:+.4f}, Allocation: {normalized_allocation * 100:.2f}%")


    final_allocations = sorted(final_allocations, key=lambda x: x[1], reverse=True)
    stop_time = time.perf_counter()

    print("time in one run vectors better pass: ", stop_time - begin_time)
    print(final_allocations)

    return final_allocations, all_closes


def optimal_picks_new(model_version, is_quantized, today=None):

    start, end = get_date_range(lookback_period, today)

    sp_tickers = get_sp500_tickers()
    if not sp_tickers:
        return None, None

    processed_tickers = [t.replace(".", "-") for t in sp_tickers]
    all_historical_data = get_historical_data(processed_tickers, start, end)
    all_close_data_full = all_historical_data["Close"]

    all_close_data = all_close_data_full.iloc[-SEQUENCE_SIZE:]
    latest_closes = all_close_data.iloc[-1]
    windows_means = all_close_data.mean(axis=0)
    windows_stds = all_close_data.std(axis=0)

    windows_stds[windows_stds <= 0] = 1
    normalized_windows = (all_close_data - windows_means) / windows_stds

    valid_tickers = normalized_windows.columns[~normalized_windows.isnull().any()]

    normalized_windows = normalized_windows[valid_tickers]
    windows_means = windows_means[valid_tickers]
    windows_stds = windows_stds[valid_tickers]
    latest_closes = latest_closes[valid_tickers]


    input_array = normalized_windows.to_numpy()
    input_tensor_batch = (
        torch.tensor(input_array, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(2)
    )

    filepath = MODEL_MAP[model_version]["model_filepath"]
    model_state_dict, config = load_model_artifacts_local(filepath)
    model = setup_pred_model(model_state_dict, config, is_quantized)

    try:
        with torch.no_grad():
            predictions_tensor = model(input_tensor_batch)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None

    predictions = pd.Series(predictions_tensor.cpu().numpy().flatten(), index=valid_tickers)

    deltas = (((predictions * windows_stds) + windows_means) - latest_closes) / latest_closes

    return deltas, all_close_data_full

