import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt


# TODO: change back all tickers to ticker it is that easy
# TODO: combine lookup_ticker and tickers????
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

        # Check for valid info and the 'shortName' key
        if info and "shortName" in info:
            # If valid, append the data to the return list
            to_ret.append({
                "ticker": ticker,
                "name": info.get("shortName", "N/A"),
                "price": info.get("currentPrice", "N/A"),
                "currency": info.get("currency", "USD"),
            })
            # Add the ticker to the list for charting
            valid_tickers_for_chart.append(ticker)
        else:
            # If not valid, print a warning and skip to the next ticker
            print(f"Warning: Could not retrieve valid info for ticker {ticker}. Skipping this ticker.")

    print(valid_tickers_for_chart)
    if valid_tickers_for_chart:
        chart, chart_data = get_chart(valid_tickers_for_chart, "1YE")
        return to_ret, chart, chart_data
    else:
        # If no valid tickers were found, return an empty list and None for the chart
        return [], None, None


def get_chart(tickers, time):
    start_time, end_time = get_date_range(time)

    is_single_ticker = len(tickers) == 1

    data = yf.download(tickers, start=start_time, end=end_time, auto_adjust=True)

    plot_data, second_data = rm_nm_and_chop2(is_single_ticker, data.copy())

    if not is_single_ticker:
        plot_data, second_data = normalize(plot_data, second_data)

    if is_single_ticker:
        title = f"{tickers[0]} Stock Price"
    else:
        title = f"Price Comparison of {", ".join(tickers)}"

    low_y, high_y = get_y_bounds(plot_data, second_data, is_single_ticker)

    if is_single_ticker:
        fig, ax = mpf.plot(
            plot_data,
            type="line",
            style="charles",
            title=title,
            returnfig=True,
            figratio=(10, 6)
        )

    if not is_single_ticker:

        print(plot_data.columns)
        fig, ax = plt.subplots()
        ax.plot(plot_data.index,plot_data["Close"], label=tickers[0])
        ax.plot(second_data.index, second_data["Close"], label=tickers[1])
        ax.set_xlabel("Date")
        ax.set_ylabel("Change (%)")
        ax.set_title(title)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        ax.set_xlim(left=plot_data.index[0], right=plot_data.index[-1])
        """
        comp_chart = mpf.make_addplot(second_data["Close"], label=tickers[1])
        combined_data = pd.DataFrame({
            tickers[0]: plot_data["Close"],
            tickers[1]: second_data["Close"]
        })
        fig, ax = mpf.plot(plot_data,
                           type="line",
                           style="charles",
                           title=title,
                           returnfig=True,
                           figratio=(10, 6),
                           #columns=["Close"]
                           addplot=comp_chart
                           )
                        
        ax = ax[0]
        print(tickers)
        ax.legend(tickers)"""

    if is_single_ticker:
        ax = ax[0]
        ax.set_ylim(low_y, high_y)

    return fig, plot_data


# TODO: complete comp chart
def rm_nm_and_chop2(is_single, df):
    plot_data = df.copy()
    second_data = None
    if not is_single:
        plot_data = df.iloc[:, ::2]
        second_data = df.iloc[:, 1::2]
        second_data.columns = second_data.columns.droplevel(1)
    plot_data.columns = plot_data.columns.droplevel(1)
    return plot_data, second_data


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
