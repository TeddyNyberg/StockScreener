import yfinance as yf
import mplfinance as mpf
import pandas as pd


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
    print(tickers)

    start_time, end_time = get_date_range(time)

    is_single_ticker = len(tickers) == 1
    print(is_single_ticker)

    data = yf.download(tickers, start=start_time, end=end_time, auto_adjust=True)

    print("data")
    print(data)

    plot_data = data.copy()
    if not is_single_ticker:
        plot_data = data.iloc[:, ::2]
        second_data = data.iloc[:, 1::2]
        second_data.columns = second_data.columns.droplevel(1)
    plot_data.columns = plot_data.columns.droplevel(1)

    if not is_single_ticker:
        for col in plot_data.columns:
            #if not plot_data[col].iloc[0] == 0:
            temp = plot_data[col].iloc[0]
            plot_data[col] = (plot_data[col] / temp - 1) * 100
        for col in second_data.columns:
            #if not second_data[col].iloc[0] == 0:
            temp = second_data[col].iloc[0]
            second_data[col] = (second_data[col] / temp - 1) * 100

    if is_single_ticker:
        title = f"{tickers} Stock Price"
    else:
        title = f"Price Comparison of {", ".join(tickers)}"

    print("MAYBE HERE??")
    print(plot_data)
    fig, ax = mpf.plot(
        plot_data,
        type="line",
        style="charles",
        title=title,
        returnfig=True,
        figratio=(10,6)
    )

    if not is_single_ticker:
        comp_chart = mpf.make_addplot(second_data)
        fig, ax = mpf.plot(plot_data,
                 type="line",
                 style="charles",
                 title=title,
                 returnfig=True,
                 figratio=(10,6),
                 addplot=comp_chart
                 )


    plot_data_no_vol = plot_data.copy()
    plot_data_no_vol = plot_data_no_vol.drop("Volume", axis=1)
    print("no vol")
    print(plot_data_no_vol)
    ax = ax[0]
    if is_single_ticker:
        low_y = plot_data_no_vol.min().min() * 0.95
        high_y = plot_data_no_vol.max().max() * 1.05
        print(low_y)
        print(high_y)
    else:
        ax.set_ylabel('Percent Change from Start (%)')
        ax.legend([ticker for ticker in tickers])
        low_y = plot_data_no_vol[0].min().min() * 0.95
        high_y = plot_data_no_vol[0].max().max() * 1.05
    ax.set_ylim(low_y, high_y)

    return fig, plot_data

# TODO: complete comp chart

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

