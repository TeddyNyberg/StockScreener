from app.search.ticker_lookup import lookup_tickers
from app.ui.window_manager import open_detail_window

# result is structured as name_and_price, chart, data
def lookup_and_open_details(ticker, display_error_func=None):
    if not ticker:
        if display_error_func:
            display_error_func("Please enter a ticker symbol.")
        return
    result = lookup_tickers(ticker)
    if not result:
        if display_error_func:
            display_error_func(f"Could not find data for ticker: {ticker}.")
        return
    open_detail_window(result)
