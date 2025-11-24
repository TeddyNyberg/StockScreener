# get S&P tickers

import requests
import pandas as pd
import io
from config import S_AND_P_URL, SP_TICKERS_TESTING


def get_sp500_tickers(btc = False, test = False):
    if test:
        return SP_TICKERS_TESTING

    r = requests.get(S_AND_P_URL)
    try:
        tables = pd.read_html(io.StringIO(r.text))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        print(tickers)
        if btc:
            tickers.append('BTC-USD')
        return tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []