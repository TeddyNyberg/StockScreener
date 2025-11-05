# get S&P tickers

import requests
import pandas as pd
import io
from config import S_AND_P_URL


def get_sp500_tickers():
    r = requests.get(S_AND_P_URL)
    print(r.status_code)
    try:
        tables = pd.read_html(io.StringIO(r.text))
        print(tables)
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []