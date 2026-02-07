# contains raw yfinance fetching
#get_stock_data, get_close ...
import pandas
# job interact w yfinance api

import yfinance as yf
from datetime import timedelta, datetime
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

    _instance = None

    CLEANUP_INTERVAL = 60
    TICKER_TTL = 600 #time to live

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LiveMarketTable, cls).__new__(cls)
            cls._instance.init_structure()
        return cls._instance

    def init_structure(self):
        self.ws = None
        self.last_day = {}
        self.listen_task = None
        self.sp_tickers = None
        self.cleanup_task = None
        self.ephemeral_tickers = {}
        self.ephemeral_prices = {}

    async def initialize(self, tickers = get_sp500_tickers()):

        try:
            self.sp_tickers = tickers.tolist()


            print(tickers, " -IN TABLE INIT-  ")
            data = yf.download(self.sp_tickers, period="5d", auto_adjust=True)["Close"].iloc[-1]
            print(data, " -IN TABLE INIT-  ")
            self.last_day = data.to_dict()
        except Exception as e:
            print(f"Except: {e}")
        await self.start_socket(self.sp_tickers)
        self.cleanup_task = asyncio.create_task(self.cleanup_loop())

    async def start_socket(self, tickers):
        print("WE'LL DO IT LIVE")
        self.ws = yf.AsyncWebSocket()
        await self.ws.subscribe(tickers)
        self.listen_task = asyncio.create_task(self.ws.listen(self.message_handler))

    async def message_handler(self, message):
        ticker = message.get('id')
        price = message.get('price')
        if ticker in self.sp_tickers:
            self.last_day[ticker] = price
        else:
            self.ephemeral_prices[ticker] = price
        #print(f"Updated {ticker}: {price}")

    async def close_socket(self):
        print("Stopping socket...")
        if self.listen_task:
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                pass

    async def add_ticker(self, ticker):
        await self.ws.subscribe([ticker])
        data = yf.download(ticker, period="5d", auto_adjust=True)["Close"].iloc[-1]
        self.ephemeral_prices[ticker] = data

    async def cur_price(self, ticker):
        if ticker in self.sp_tickers:
            return self.last_day[ticker]
        if ticker in self.ephemeral_tickers:
            self.ephemeral_tickers[ticker] = datetime.now()
            return self.ephemeral_tickers[ticker]
        self.ephemeral_tickers[ticker] = datetime.now()
        await self.add_ticker(ticker)
        return self.ephemeral_prices[ticker]

    async def cleanup_loop(self):
        while True:

            await asyncio.sleep(self.CLEANUP_INTERVAL)
            print("====")
            print(self.last_day.keys())
            print("----")
            print(self.ephemeral_tickers.keys())
            print("====")
            now = datetime.now()
            limit = now - timedelta(seconds=self.TICKER_TTL)
            tickers_to_remove = []
            for ticker, last_access in list(self.ephemeral_tickers.items()):
                if last_access < limit:
                    tickers_to_remove.append(ticker)
            if tickers_to_remove:
                print(f"Unsubscribing from stale tickers: {tickers_to_remove}")

                await self.ws.unsubscribe(tickers_to_remove)
                for t in tickers_to_remove:
                    del self.ephemeral_tickers[t]
                    del self.ephemeral_prices[t]

    def get_snapshot(self):
        return pandas.Series(self.last_day)