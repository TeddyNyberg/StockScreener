import yfinance as yf
import mplfinance as mpf
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
            "chart": get_chart(ticker, "1YE")  #D, ME, YE
        }

    except Exception as e:
        print(f"Error looking up {ticker}: {e}")
        return None


def get_chart(ticker, time):
    start_time, end_time = get_date_range(time)

    data = yf.download(ticker, start=start_time, end=end_time, auto_adjust=True)
    data.columns = data.columns.droplevel(1)
    print(data)

    fig, axlist = mpf.plot(
        data,
        type='line', # or candle or line
        style='charles',
        title=f"{ticker} Stock Price",
        returnfig=True
    )

    low_price = data['Low'].min()
    high_price = data['High'].max()

    padding = (high_price - low_price) * 0.1

    ax = axlist[0]
    ax.set_ylim(low_price - padding, high_price + padding)
    fig.data = [data]

    return fig


def get_date_range(time):  # must be all caps
    today = pd.Timestamp.today().normalize()
    if time == "YTD":
        start = pd.Timestamp(year=today.year, month=1, day=1)
    elif time == "MAX":
        start = pd.Timestamp(year=1950, month=1, day=1)
    else:
        start = today - pd.tseries.frequencies.to_offset(time)
    return start, today


def get_yfticker(ticker):
    return yf.Ticker(ticker)

def get_financial_metrics(ticker):
    # TODO: scale financials and return the scale with it too
    # TODO: remove things I dont care abt??
    return yf.Ticker(ticker).financials

