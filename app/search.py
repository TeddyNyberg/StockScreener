import yfinance as yf
import mplfinance as mpf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
def lookup_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or "shortName" not in info:
            return None

        return {
            "ticker": ticker,
            "name": info.get("shortName", "N/A"),
            "price": info.get("currentPrice", "N/A"),
            "currency": info.get("currency", "USD"),
            "chart": get_chart(ticker, "1D")
        }

    except Exception as e:
        print(f"Error looking up {ticker}: {e}")
        return None


def get_chart(ticker, time):
    start_time, end_time = get_date_range(ticker,time)

    data = yf.download(ticker, start=start_time, end=end_time, auto_adjust=True)
    fig, axlist = mpf.plot(
        data,
        type='candle',
        style='charles',  # or other style you like
        title=f"{ticker} Stock Price",
        returnfig=True
    )
    return fig



def get_date_range(ticker, time):  # must be all caps
    today = pd.Timestamp.today().normalize()
    if time == "YTD":
        start = pd.Timestamp(year=today.year, month=1, day=1)
    else:
        start = today - pd.tseries.frequencies.to_offset(time)

    count = 0
    while True:
        data = yf.download(ticker, start=start - pd.Timedelta(days=count), end=today, auto_adjust=True)
        #print("Here")
        #print(data)
        count += 1
        if not data.empty: #and "Open" in data.columns:
            #data = data.dropna(subset=["Open"])
            #if not data.empty and "Open" in data.columns:
            #    print(count)
            return data.index.min().date(), data.index.max().date()

