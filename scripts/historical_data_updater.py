import yfinance as yf
import os
from backend.app.data.ticker_source import get_sp500_tickers
from backend.app.ml_logic.strategy import get_all_volatilities_np
from backend.app.utils import get_date_range
from backend.app.config import *
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PARQ_PATH = os.path.join(PROJECT_ROOT, SP_DATA_CACHE)
VOL_PATH = os.path.join(PROJECT_ROOT, VOL_DATA_CACHE)


def update_cache():
    tickers = get_sp500_tickers()

    print("Downloading historical data...")
    start, end = get_date_range("1Y", None)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close_data = data["Close"].iloc[-125:]
    volatility_array = get_all_volatilities_np(close_data.to_numpy())
    np.save(VOL_PATH, volatility_array)
    data["Close"].to_parquet(PARQ_PATH)   #dropped .iloc[-(SEQUENCE_SIZE - 1):]
    print("Cache updated.")


if __name__ == "__main__":
    update_cache()
