import yfinance as yf
import os
from app.data.ticker_source import get_sp500_tickers
from app.ml_logic.strategy import get_all_volatilities_np
from app.utils import get_date_range
from config import *
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PARQ_PATH = os.path.join(PROJECT_ROOT, SP_DATA_CACHE)
VOL_PATH = os.path.join(PROJECT_ROOT, VOL_DATA_CACHE)

def update_cache():
    tickers = get_sp500_tickers()

    print("Downloading historical data...")
    start, end = get_date_range(lookback_period, None)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close_data = data["Close"]
    volatility_array = get_all_volatilities_np(close_data.to_numpy())
    np.save(VOL_PATH, volatility_array)
    data["Close"].iloc[-(SEQUENCE_SIZE - 1):].to_parquet(PARQ_PATH)
    print("Cache updated.")


if __name__ == "__main__":
    update_cache()



"""
mus_arr, valid_tickers, all_vol_data = (
        optimal_picks(model_version=model_version, is_quantized=is_quantized, today=end))
    all_most_recent_closes = all_vol_data.iloc[-1]

    if mus_arr is None or len(mus_arr) == 0:
        print("No predictions available to calculate Kelly bets.")
        return None
    volatility_series = get_all_volatilities(all_vol_data)

    sigma_squareds_aligned = volatility_series.reindex(valid_tickers, fill_value=-1)
    sigma_squareds_array = sigma_squareds_aligned.to_numpy()

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
"""
