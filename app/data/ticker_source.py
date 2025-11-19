# get S&P tickers

import requests
import pandas as pd
import io
from config import S_AND_P_URL

# POSSIBLE DATA LEAKAGE
# TESTED USING CURRENT S&P COMPANIES
# NOT COMPANIES THAT WERE IN THE S&P THEN
def get_sp500_tickers(btc = False):
    r = requests.get(S_AND_P_URL)
    try:
        tables = pd.read_html(io.StringIO(r.text))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        if btc:
            tickers.append('BTC-USD')
        return tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []