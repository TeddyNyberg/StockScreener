import pandas as pd
from backend.app.utils import get_date_range
from backend.app.data.yfinance_fetcher import get_historical_data
from backend.app.data.nyberg_fetcher import get_nyberg_data
from backend.config import *
import numpy as np


_cache = {}  # global cache: {ticker: {("start", "end"): DataFrame}}


def rm_nm(data_list):
    for df in data_list:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df .columns.droplevel(1)
    return data_list


# takes (list)tickers and (string)time, returns df1, df2
# issue yf returns dif structure if needing 1 vs many,
# so need to remove names if many.
# also handle nyberg

def get_yfdata_cache(tickers: list[str], time: str = None, normalize=True):
    print("CALLED GET DATA")
    print(tickers)
    print(time)
    print("---------")
    start_time, end_time = get_date_range(time, normalize=normalize)
    results = []

    for ticker in tickers:
        if ticker is None:
            results.append(None)
            continue

        if ticker.startswith("NYBERG"):
            results.append(get_nyberg_data(time, ticker))
            continue

        if ticker not in _cache:
            df = get_historical_data(ticker, start_time, end_time)

            if normalize:
                _cache[ticker] = {"range": (start_time, end_time),
                                "data": df}
        else:
            cache_start, cache_end = _cache[ticker]["range"]
            cached_df = _cache[ticker]["data"]

            if start_time >= cache_start and end_time <= cache_end:
                df = cached_df.loc[start_time:end_time]

            else:
                new_start = min(start_time, cache_start)
                new_end = max(end_time, cache_end)
                new_df = get_historical_data(ticker, new_start, new_end)
                if normalize:
                    _cache[ticker]["range"] = (new_start, new_end)
                    _cache[ticker]["data"] = new_df
                df = new_df.loc[start_time:end_time]
        results.append(df)

    return rm_nm(results)


def get_volatility_cache():
    return np.load(VOL_DATA_CACHE)

def get_cached_49days():
    return pd.read_parquet(SP_DATA_CACHE)
