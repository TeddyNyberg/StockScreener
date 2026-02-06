# get S&P tickers

import requests
import pandas as pd
import io
from backend.app.config import S_AND_P_URL, SP_TICKER_CACHE
import csv
import os

# POSSIBLE DATA LEAKAGE
# TESTED USING CURRENT S&P COMPANIES
# NOT COMPANIES THAT WERE IN THE S&P THEN
def get_sp500_tickers():
    if os.path.exists(SP_TICKER_CACHE):
        try:
            with open(SP_TICKER_CACHE, 'r') as f:
                reader = csv.reader(f)
                tickers = next(reader)
                if tickers:
                    return tickers
        except Exception as e:
            print(f"Error w cache, fetching from internet: {e}")
    r = requests.get(S_AND_P_URL)
    try:
        # Do not update cache here
        # if it misses and gets here, and you do update cache, it will never miss again,
        # and you will be stuck w outdated tickers
        # the script should run every day and update the cache
        tables = pd.read_html(io.StringIO(r.text))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        processed_tickers = [t.replace(".", "-") for t in tickers]
        return processed_tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []