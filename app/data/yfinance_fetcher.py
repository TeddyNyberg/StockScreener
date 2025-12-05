# contains raw yfinance fetching
#get_stock_data, get_close ...

# job interact w yfinance api

import yfinance as yf
from datetime import timedelta
import asyncio
import pandas as pd
from config import *


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




class LiveMarketTable:
    def __init__(self):
        self.ws = None
        self.last_day = None
        self.listen_task = None
        try:
            df = pd.read_parquet(SP_DATA_CACHE)
            last_day = df.iloc[-1]
            self.last_day = last_day
        except Exception:
            print("File not found")


    async def start_socket(self, tickers):
        print("WE'LL DO IT LIVE")
        self.ws = yf.AsyncWebSocket()
        await self.ws.subscribe(tickers)
        self.listen_task = asyncio.create_task(self.ws.listen(self.message_handler))

    async def message_handler(self, message):
        ticker = message.get('id')
        price = message.get('price')
        self.last_day.at[ticker] = price
        print(f"Updated {ticker}: {price}")

    async def close_socket(self):
        print("Stopping socket...")
        self.listen_task.cancel()
        await self.ws.close()
        return self.last_day