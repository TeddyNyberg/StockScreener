from backend.config import *
import requests
import csv
import pandas as pd
import io
import os

def fetch_and_cache_tickers():


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    CSV_PATH = os.path.join(PROJECT_ROOT, SP_TICKER_CACHE)

    try:
        r = requests.get(S_AND_P_URL)
        new_tickers = extract_tickers_from_response(r)
        processed_tickers = [t.replace(".", "-") for t in new_tickers]
        #sp_cache = "../" + SP_CACHE_FILE
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
    fetch_and_cache_tickers()


