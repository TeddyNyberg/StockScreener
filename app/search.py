import yfinance as yf
import mplfinance as mpf
import pandas as pd
from pandas.tseries.offsets import DateOffset, Day, MonthEnd

_cache = {}  # global cache: {ticker: {("start", "end"): DataFrame}}


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
        stock = yf.Ticker(ticker)
        info = stock.info

        if info and "shortName" in info:
            to_ret.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "price": info.get("currentPrice", "N/A"),
                "currency": info.get("currency", "USD"),
            })
            valid_tickers_for_chart.append(ticker)
        else:
            print(f"Warning: Could not retrieve valid info for ticker {ticker}. Skipping this ticker.")

    if valid_tickers_for_chart:
        chart, chart_data = get_chart(valid_tickers_for_chart, "1Y")
        return to_ret, chart, chart_data
    else:
        return [], None, None


# limitation of get_chart: it only works with 1 or 2 tickers.
# also all supporting functions only work with 1 or 2 tickers.
# its all hardcoded
# TODO: allow adding more tickers for comparison
def get_chart(tickers, time):
    is_single_ticker = len(tickers) == 1 or tickers[1] is None

    plot_data, second_data = get_yfdata_cache(tickers, time)

    title = get_title(tickers)

    if not is_single_ticker:
        plot_data, second_data = normalize(plot_data, second_data)

    plot_kwargs = {
        "type": "line",
        "style": "charles",
        "title": title,
        "returnfig": True,
        "figratio": (10, 6)
    }
    if not is_single_ticker:
        comp_chart = mpf.make_addplot(second_data["Close"], label=tickers[1], secondary_y=False)
        plot_kwargs["addplot"] = comp_chart

    fig, ax = mpf.plot(plot_data, **plot_kwargs)
    ax = ax[0]
    if not is_single_ticker:
        ax.set_ylabel("Change (%)")
        ax.legend(tickers)
    else:
        low_y, high_y = get_y_bounds(plot_data, second_data, is_single_ticker)
        ax.set_ylim(low_y, high_y)

    return fig, plot_data


def rm_nm(df1, df2=None):
    if isinstance(df1.columns, pd.MultiIndex):
        df1.columns = df1.columns.droplevel(1)
    if df2 is not None and isinstance(df2.columns, pd.MultiIndex):
        df2.columns = df2.columns.droplevel(1)
    return df1, df2


def get_title(tickers):
    title = f"{tickers[0]} Stock Price"
    if len(tickers) > 1 and tickers[1] is not None:
        title = f"Price Comparison of {", ".join(tickers)}"
    return title


def get_yfdata_cache(tickers, time):
    start_time, end_time = get_date_range(time)
    results = []

    for ticker in tickers:
        if ticker is None:
            results.append(None)
            continue

        if ticker not in _cache:
            df = yf.download(ticker, start=start_time, end=end_time, auto_adjust=True)
            _cache[ticker] = {"range": (start_time, end_time),
                              "data": df}
        else:
            cache_start, cache_end = _cache[ticker]["range"]
            cached_df = _cache[ticker]["data"]

            if start_time >= cache_start and end_time <= cache_end:
                df = cached_df.loc[start_time:end_time]

            else:
                new_start = min(start_time, cache_start)
                new_end = max(end_time, cache_end)

                new_df = yf.download(ticker, start=new_start, end=new_end, auto_adjust=True)
                _cache[ticker]["range"] = (new_start, new_end)
                _cache[ticker]["data"] = new_df
                df = new_df.loc[start_time:end_time]
        results.append(df)

    if len(results) == 1:
        results.append(None)
    return rm_nm(results[0], results[1])


def normalize(df1, df2):
    df1 = (df1.div(df1.iloc[0]) - 1) * 100
    df2 = (df2.div(df2.iloc[0]) - 1) * 100
    return df1, df2


def get_y_bounds(df1, df2, is_single):
    df1_no_vol = df1.drop("Volume", axis=1)
    low_y = df1_no_vol.min().min() * 0.95
    high_y = df1_no_vol.max().max() * 1.05
    if not is_single:
        df2_no_vol = df2.drop("Volume", axis=1)
        low_y_2 = df2_no_vol.min().min() * 0.95
        high_y_2 = df2_no_vol.max().max() * 1.05
        low_y = min(low_y, low_y_2)
        high_y = max(high_y, high_y_2)
    return low_y, high_y


def get_date_range(time):  # must be all caps
    today = pd.Timestamp.today().normalize()

    offsets = {
        "1D": Day(1),
        "5D": Day(5),
        "1M": DateOffset(months=1),
        "3M": DateOffset(months=3),
        "6M": DateOffset(months=6),
        "1Y": DateOffset(years=1),
        "3Y": DateOffset(years=3),
        "5Y": DateOffset(years=5),
        "YTD": DateOffset(year=today.year, month=1, day=1)
    }
    if time == "MAX":
        start = pd.Timestamp(year=1950, month=1, day=1)
    else:
        start = today - offsets[time]
    return start, today


def get_yfticker(ticker):
    return yf.Ticker(ticker)


def get_financial_metrics(ticker):
    # TODO: scale financials and return the scale with it too
    # TODO: remove things I dont care abt??
    return yf.Ticker(ticker).financials


def get_balancesheet(ticker):
    return yf.Ticker(ticker).balancesheet


def get_info(ticker):
    print(yf.Ticker(ticker).info)
    return yf.Ticker(ticker).info

# def getxx(ticker):
#    return yf.Ticker(ticker).
