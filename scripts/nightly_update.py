import sys
import yfinance as yf
from backend.app import get_sp500_tickers
from backend.app.ml_logic import get_all_volatilities_np
from backend.app import get_date_range
import numpy as np
from config import *
import requests
import csv
import pandas as pd
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PARQ_PATH = os.path.join(PROJECT_ROOT, SP_DATA_CACHE)
VOL_PATH = os.path.join(PROJECT_ROOT, VOL_DATA_CACHE)
CSV_PATH = os.path.join(PROJECT_ROOT, SP_TICKER_CACHE)

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

def fetch_and_cache_tickers():
    try:
        r = requests.get(S_AND_P_URL)
        new_tickers = extract_tickers_from_response(r)
        processed_tickers = [t.replace(".", "-") for t in new_tickers]
        # sp_cache = "../" + SP_CACHE_FILE
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(processed_tickers)

        print(f"Successfully updated {SP_TICKER_CACHE} with {len(processed_tickers)} tickers.")
        return True
    except Exception as e:
        print(f"Error fetching or writing tickers: {e}")
        return False

def extract_tickers_from_response(r):
    try:
        tables = pd.read_html(io.StringIO(r.text))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []


if __name__ == "__main__":
    print("--- Starting Nightly Update ---")

    ok = fetch_and_cache_tickers()
    if not ok:
        print("CRITICAL: Ticker update failed. Aborting history update to prevent data corruption.")
        sys.exit(1)
    try:
        update_cache()
    except Exception as e:
        print(f"CRITICAL: History update crashed: {e}")
        sys.exit(1)
    print("--- Nightly Update Complete ---")

