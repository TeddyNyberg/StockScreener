import pandas as pd
from app.utils import get_date_range
from app.data.yfinance_fetcher import get_historical_data
from app.data.nyberg_fetcher import get_nyberg_data



_cache = {}  # global cache: {ticker: {("start", "end"): DataFrame}}

def rm_nm(df1, df2=None):
    if isinstance(df1.columns, pd.MultiIndex):
        df1.columns = df1.columns.droplevel(1)
    if df2 is not None and isinstance(df2.columns, pd.MultiIndex):
        df2.columns = df2.columns.droplevel(1)
    return df1, df2



def get_yfdata_cache(tickers, time):
    start_time, end_time = get_date_range(time)
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
                _cache[ticker]["range"] = (new_start, new_end)
                _cache[ticker]["data"] = new_df
                df = new_df.loc[start_time:end_time]
        results.append(df)

    if len(results) == 1:
        results.append(None)
    return rm_nm(results[0], results[1])

