from app.data.yfinance_fetcher import get_historical_data
from app.data.ticker_source import get_sp500_tickers
from app.data.preprocessor_utils import prepare_data_for_prediction
from app.ml_logic.pred_models.only_close_model import pred_next_day_no_ticker, fine_tune_model, setup_pred_model
from app.utils import get_date_range
from app.ml_logic.model_loader import load_model_artifacts, save_model_artifacts
from config import *
import numpy as np
import torch
import pandas as pd



def optimal_picks(model_version, is_quantized, today=None):
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
    model_state_dict, config = load_model_artifacts(filepath)
    model, _ = setup_pred_model(model_state_dict, config, is_quantized)

    try:
        with torch.no_grad():
            predictions_tensor = model(input_tensor_batch)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None

    predictions = pd.Series(predictions_tensor.cpu().numpy().flatten(), index=valid_tickers)
    deltas = (((predictions * windows_stds) + windows_means) - latest_closes) / latest_closes

    return deltas, all_close_data_full



def calculate_kelly_allocations(model_version, is_quantized, end=None):

    mus, all_vol_data = optimal_picks(model_version, is_quantized, end)
    all_closes = all_vol_data.iloc[-1]

    if mus is None or mus.Empty:
        print("No predictions available to calculate Kelly bets.")
        return None
    volatility_series = get_all_volatilities(all_vol_data)
#
    tickers = []
    mu_list = []
    s2_list = []

    for ticker, predicted_delta in predictions:
        sigma_squared = volatility_series.get(ticker, -1)
        tickers.append(ticker)
        mu_list.append(predicted_delta)
        s2_list.append(sigma_squared)
#
    mus = pd.Series(mu_list, index=tickers)
    sigma_squareds = pd.Series(s2_list, index=tickers)

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

    return final_allocations, all_closes

def predict_single_ticker(ticker):
    model_state_dict, config = load_model_artifacts()

    start, end = get_date_range("3M")
    data = get_historical_data(ticker, start, end)
    close_data = data["Close"]
    input_tensor, _, mean, std = prepare_data_for_prediction(close_data)
    prediction = pred_next_day_no_ticker(input_tensor, model_state_dict, config, mean, std)

    print(f"Predicted value: {prediction}")
    return prediction

def get_all_volatilities(all_data):
    returns_df = all_data.pct_change(fill_method=None)
    returns_df = returns_df.dropna(how='all', axis=0)
    returns_df = returns_df.dropna(how='all', axis=1)

    daily_variance = returns_df.var()
    annualized_variance_series = daily_variance * 252 #252 trading days

    return annualized_variance_series



def tune(filepath, date):
    model_dict, config = load_model_artifacts(filepath)
    start, end = get_date_range("3M", date)

    snp = get_sp500_tickers()
    if not snp:
        return False
    list_of_df = []
    for ticker in snp:
        list_of_df.append(get_historical_data(ticker, start, end))
    new_model_dict, new_config = fine_tune_model(model_dict, config, list_of_df)
    save_model_artifacts(new_model_dict, new_config, filepath)
    return True
