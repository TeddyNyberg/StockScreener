from backend.app.data.yfinance_fetcher import get_info
from backend.app.data.nyberg_fetcher import get_nyberg_price, get_nyberg_name
from backend.app.search.charting import get_chart
from backend.config import *

# returns 3 things, normally called result, chart, data
# result = list of quaduple{ticker, price, currency, name}for each stock
# chart = figure
# data = data use to make chart
def lookup_tickers(tickers):
    to_ret = []
    valid_tickers_for_chart = []

    if isinstance(tickers, str):
        tickers = [tickers]

    # by now tickers is a list regardless of how it comes in
    for ticker in tickers:
        if ticker.startswith("NYBERG"):
            to_ret.append({
                "ticker": ticker,
                "name": get_nyberg_name(ticker),
                "price": get_nyberg_price(ticker),
                "currency": "USD"
            })
            valid_tickers_for_chart.append(ticker)
            continue


        #TODO: if yfinance down, this all breaks. Fix case where on init, info not returned
        try:
            info = get_info(ticker)
            to_ret.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "price": info.get("currentPrice", "N/A"),
                "currency": info.get("currency", "USD"),
            })
            valid_tickers_for_chart.append(ticker)
        except:
            print(f"no info for {ticker}")

    if valid_tickers_for_chart:
        chart, chart_data = get_chart(valid_tickers_for_chart, DEFAULT_CHART_TIME)
        return to_ret, chart, chart_data
    else:
        return [], None, None