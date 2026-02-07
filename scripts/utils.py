import io
import pandas as pd


def extract_tickers_from_response(r):
    try:
        tables = pd.read_html(io.StringIO(r.text))
        table = tables[0]
        tickers = table['Symbol'].tolist()
        return tickers
    except Exception as e:
        print(f"Error scraping S&P 500 tickers: {e}")
        return []
