from app.data.yfinance_fetcher import get_historical_data
from app.data.ticker_source import get_sp500_tickers
from app.data.preprocessor_utils import prepare_data_for_prediction
from app.ml_logic.prediction_models.only_close_model import pred_next_day_no_ticker, fine_tune_model, setup_pred_model
from app.utils import get_date_range
from app.ml_logic.model_loader import load_model_artifacts, save_model_artifacts
from config import *
import numpy as np
import torch
import time

def optimal_picks(model_version, is_quantized, today=None):

    start, end = get_date_range(lookback_period, today)

    sp_tickers = get_sp500_tickers()
    if not sp_tickers:
        return None, None

    all_historical_data = get_historical_data(sp_tickers, start, end)
    all_close_data_full = all_historical_data["Close"]

    all_close_data = all_close_data_full.iloc[-SEQUENCE_SIZE:].to_numpy()
    latest_closes = all_close_data[-1,:]
    windows_means = np.mean(all_close_data, axis=0)
    windows_stds = np.std(all_close_data, axis=0, ddof=1)

    windows_stds[windows_stds <= 0] = 1
    normalized_windows = (all_close_data - windows_means) / windows_stds

    valid_mask = ~np.isnan(normalized_windows).any(axis=0)

    normalized_windows = normalized_windows[:, valid_mask]
    windows_means = windows_means[valid_mask]
    windows_stds = windows_stds[valid_mask]
    latest_closes = latest_closes[valid_mask]
    all_close_data_full = all_close_data_full.loc[:, valid_mask]

    valid_tickers = np.array(all_close_data_full.columns.tolist())

    input_tensor_batch = (
        torch.tensor(normalized_windows, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(2)
    )

    filepath = MODEL_MAP[model_version]["model_filepath"]
    model_state_dict, config = load_model_artifacts(filepath)
    model = setup_pred_model(model_state_dict, config, is_quantized)

    device = next(model.parameters()).device
    input_tensor_batch = input_tensor_batch.to(device)

    try:
        with torch.no_grad():
            predictions_tensor = model(input_tensor_batch)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None

    predictions = predictions_tensor.cpu().numpy().flatten()
    deltas = (((predictions * windows_stds) + windows_means) - latest_closes) / latest_closes

    return deltas, valid_tickers, all_close_data_full


def calculate_kelly_allocations(model_version, is_quantized, end=None, only_largest=False):

    mus_arr, valid_tickers, all_vol_data = (
        optimal_picks(model_version=model_version, is_quantized=is_quantized, today=end))
    all_most_recent_closes = all_vol_data.iloc[-1]

    if mus_arr is None or len(mus_arr) == 0:
        print("No predictions available to calculate Kelly bets.")
        return None

    sigma_squareds_array = get_all_volatilities_np(all_vol_data.to_numpy())

    valid_mask = (sigma_squareds_array > 0) & (~np.isnan(sigma_squareds_array)) & (~np.isnan(mus_arr))

    final_tickers = valid_tickers[valid_mask]
    final_mus = mus_arr[valid_mask]
    final_sigma_squareds = sigma_squareds_array[valid_mask]

    kelly_fraction = (final_mus - RISK_FREE_RATE) / final_sigma_squareds
    allocation = kelly_fraction * HALF_KELLY_MULTIPLIER
    allocation = np.clip(allocation, 0.0, 1.0)

    positive_mask = allocation > 0.0

    final_allocation_values = allocation[positive_mask]
    final_tickers_really = final_tickers[positive_mask]
    final_mus_to_trade = final_mus[positive_mask]


    if not only_largest:
        total_allocation = final_allocation_values.sum()
        normalization_factor = 1.0 / total_allocation if total_allocation > 1.0 else 1.0
        normalized_allocations = final_allocation_values * normalization_factor
    else:
        normalized_allocations = final_allocation_values
        normalization_factor = 1.0
        total_allocation = 1

    final_allocations = [
        (ticker, normalized_allocation, mu)
        for ticker, normalized_allocation, mu in
        zip(final_tickers_really, normalized_allocations, final_mus_to_trade)
    ]

    sorted_allocations = sorted(final_allocations, key=lambda x: x[1], reverse=True)

    if only_largest:
        final_allocations = []
        summation = 0

        for ticker, raw_allocation, mu in sorted_allocations:
            remaining_cap = 1 - summation
            if remaining_cap <= 1e-6:
                break

            allocation_to_use = min(raw_allocation, remaining_cap)

            final_allocations.append((ticker, allocation_to_use, mu))
            summation += allocation_to_use
    else:
        final_allocations = sorted_allocations


    print("\n--- Continuous Kelly-Based Position Sizing ---")
    print(f"Total Unnormalized Allocation: {total_allocation * 100:.2f}%")
    print(f"Normalization Factor (if > 100%): {normalization_factor:.4f}")


    for ticker, normalized_allocation, mu in final_allocations:
        print(f"Stock: {ticker}, μ: {mu:+.4f}, Allocation: {normalized_allocation * 100:.2f}%")

    return final_allocations, all_most_recent_closes


def get_all_volatilities_np(data_array):
    prev_prices = data_array[:-1]
    curr_prices = data_array[1:]
    returns = (curr_prices - prev_prices) / prev_prices
    daily_variance = np.nanvar(returns, axis=0, ddof=1)
    annualized_variance = daily_variance * 252 #252 trading days

    return annualized_variance


def tune(version, date):
    filepath = MODEL_MAP[version]["model_filepath"]
    model_dict, config = load_model_artifacts(filepath)
    start, end = get_date_range("3M", date)

    snp = get_sp500_tickers()
    if not snp:
        return False

    all_data = get_historical_data(snp, start, end)
    all_closes = all_data["Close"]

    new_model_dict, new_config = fine_tune_model(model_dict, config, all_closes)
    save_model_artifacts(new_model_dict, new_config, filepath)
    return True




# not generalizible, only model A
def fastest_kelly(data, model, vol_data, tickers):

    mus_arr, optimal_val_mask = fastest_optimal(data, model)

    if mus_arr is None or len(mus_arr) == 0:
        print("No predictions available to calculate Kelly bets.")
        return None

    vol_data = vol_data[optimal_val_mask]

    valid_mask = (vol_data > 0) & (~np.isnan(vol_data)) & (~np.isnan(mus_arr))

    valid_tickers = tickers[optimal_val_mask]
    final_tickers = valid_tickers[valid_mask]
    final_mus = mus_arr[valid_mask]
    final_sigma_squareds = vol_data[valid_mask]

    kelly_fraction = (final_mus - RISK_FREE_RATE) / final_sigma_squareds
    allocation = kelly_fraction * HALF_KELLY_MULTIPLIER
    allocation = np.clip(allocation, 0.0, 1.0)

    positive_mask = allocation > 0.0

    final_allocation_values = allocation[positive_mask]
    final_tickers_really = final_tickers[positive_mask]
    final_mus_to_trade = final_mus[positive_mask]


    total_allocation = final_allocation_values.sum()
    normalization_factor = 1.0 / total_allocation if total_allocation > 1.0 else 1.0
    normalized_allocations = final_allocation_values * normalization_factor

    final_allocations = [
        (ticker, normalized_allocation, mu)
        for ticker, normalized_allocation, mu in
        zip(final_tickers_really, normalized_allocations, final_mus_to_trade)
    ]

    sorted_allocations = sorted(final_allocations, key=lambda x: x[1], reverse=True)


    print("\n--- Continuous Kelly-Based Position Sizing ---")
    print(f"Total Unnormalized Allocation: {total_allocation * 100:.2f}%")
    print(f"Normalization Factor (if > 100%): {normalization_factor:.4f}")

    for ticker, normalized_allocation, mu in sorted_allocations:
        print(f"Stock: {ticker}, μ: {mu:+.4f}, Allocation: {normalized_allocation * 100:.2f}%")

    return sorted_allocations

def fastest_optimal(all_close_data, model):

    latest_closes = all_close_data[-1, :]
    windows_means = np.mean(all_close_data, axis=0)
    windows_stds = np.std(all_close_data, axis=0, ddof=1)

    windows_stds[windows_stds <= 0] = 1
    normalized_windows = (all_close_data - windows_means) / windows_stds

    valid_mask = ~np.isnan(normalized_windows).any(axis=0)

    normalized_windows = normalized_windows[:, valid_mask]
    windows_means = windows_means[valid_mask]
    windows_stds = windows_stds[valid_mask]
    latest_closes = latest_closes[valid_mask]


    input_tensor_batch = (
        torch.tensor(normalized_windows, dtype=torch.float32)
        .transpose(0, 1)
        .unsqueeze(2)
    )


    device = next(model.parameters()).device
    input_tensor_batch = input_tensor_batch.to(device)

    try:
        with torch.no_grad():
            predictions_tensor = model(input_tensor_batch)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None

    predictions = predictions_tensor.cpu().numpy().flatten()
    deltas = (((predictions * windows_stds) + windows_means) - latest_closes) / latest_closes

    return deltas, valid_mask