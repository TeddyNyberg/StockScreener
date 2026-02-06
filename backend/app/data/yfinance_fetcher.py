# contains raw yfinance fetching
#get_stock_data, get_close ...

# job interact w yfinance api

import yfinance as yf
from datetime import timedelta
import asyncio
from backend.app.data.ticker_source import get_sp500_tickers
import numpy as np


# this may lead to errors, understand some may want ticker col some dont
def get_historical_data(ticker, start, end, ticker_col = False):
    stock_data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if ticker_col:
        stock_data['Ticker'] = ticker
    return stock_data


# TODO: this break all old pyside ui
def get_info(ticker_list: list[str]):
    response = {}
    for ticker in ticker_list:
        info = yf.Ticker(ticker).info
        response[ticker] = info

    return response

# Ensure `date` is a string like '2024-10-15'
def get_close_on(ticker, date):
    data = yf.download(ticker, start=date, end=date + timedelta(days=1), progress=False, auto_adjust=True)
    if not data.empty:
        return data["Close"]
    else:
        return None

def get_price(ticker):
    return yf.Ticker(ticker).info.get("regularMarketPrice")


# TODO: this break all old pyside ui
def get_financial_metrics(ticker_list: list[str]):
    response = {}
    for ticker in ticker_list:
        df = yf.Ticker(ticker).financials
        df = df.replace([np.nan, np.inf, -np.inf], None)
        info = df.reset_index().to_dict(orient="records")
        response[ticker] = info

    return response


# TODO: this break all old pyside ui
def get_balancesheet(ticker_list: list[str]):
    response = {}
    for ticker in ticker_list:
        df = yf.Ticker(ticker).balancesheet
        df = df.replace([np.nan, np.inf, -np.inf], None)
        info = df.reset_index().to_dict(orient="records")
        response[ticker] = info
    return response


class LiveMarketTable:
    def __init__(self):
        self.ws = None
        self.last_day = None
        self.listen_task = None
        try:
            sp_tickers = get_sp500_tickers()
            data = yf.download(sp_tickers, period="5d", auto_adjust=True)["Close"].iloc[-1]
            self.last_day = data
        except Exception as e:
            print(f"Except: {e}")


    async def start_socket(self, tickers):
        print("WE'LL DO IT LIVE")
        self.ws = yf.AsyncWebSocket()
        await self.ws.subscribe(tickers)
        self.listen_task = asyncio.create_task(self.ws.listen(self.message_handler))

    async def message_handler(self, message):
        ticker = message.get('id')
        price = message.get('price')
        self.last_day.at[ticker] = price
        #print(f"Updated {ticker}: {price}")

    async def close_socket(self):
        print("Stopping socket...")
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass

    def get_snapshot(self):
        return self.last_day.copy()