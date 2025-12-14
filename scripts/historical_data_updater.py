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
