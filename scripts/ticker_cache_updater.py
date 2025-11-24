from config import *
import requests
import csv
import pandas as pd
import io


def fetch_and_cache_tickers():
    try:
        r = requests.get(S_AND_P_URL)
        new_tickers = extract_tickers_from_response(r)
        sp_cache = "../" + SP_CACHE_FILE
        with open(sp_cache, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(new_tickers)

        print(f"Successfully updated {SP_CACHE_FILE} with {len(new_tickers)} tickers.")

    except Exception as e:
        print(f"Error fetching or writing tickers: {e}")

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


