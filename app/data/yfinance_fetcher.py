# contains raw yfinance fetching
#get_stock_data, get_close ...

# job interact w yfinance api

import yfinance as yf
from datetime import timedelta


# this may lead to errors, understand some may want ticker col some dont
def get_historical_data(ticker, start, end, ticker_col = False):
    stock_data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if ticker_col:
        stock_data['Ticker'] = ticker
    return stock_data

def get_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

# Ensure `date` is a string like '2024-10-15'
def get_close_on(ticker, date):
    data = yf.download(ticker, start=date, end=date + timedelta(days=1), progress=False, auto_adjust=True)
    if not data.empty:
        return data["Close"]
    else:
        return None

def get_price(ticker):
    return yf.Ticker(ticker).info.get("regularMarketPrice")

def get_financial_metrics(ticker):
    # TODO: scale financials and return the scale with it too
    # TODO: remove things I dont care abt??
    return yf.Ticker(ticker).financials


def get_balancesheet(ticker):
    return yf.Ticker(ticker).balancesheet



