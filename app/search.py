import yfinance as yf
import mplfinance as mpf
import pandas as pd



# TODO: combine lookup_ticker and tickers????
def lookup_tickers(tickers):
    to_ret = []
    valid_tickers_for_chart = []


    if isinstance(tickers, str):
        tickers = [tickers]

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

    if isinstance(tickers, str):
        ticker_list = [tickers]
        is_single_ticker = True
    else:
        ticker_list = list(tickers)
        is_single_ticker = len(ticker_list) == 1

    data = yf.download(ticker_list, start=start_time, end=end_time, auto_adjust=True)
    print(data)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    print(data)

    if is_single_ticker:
        plot_data = data
        title = f"{ticker_list[0]} Stock Price"
        type = 'line'
    else:
        close_data = data['Adj Close']

        plot_data = close_data / close_data.iloc[0] * 100

        title = f"Price Comparison of {', '.join(ticker_list)}"
        type = 'line'
    print("hello hombre")
    print(plot_data)
    fig, axlist = mpf.plot(
        plot_data,
        type=type,
        style='charles',
        title=title,
        returnfig=True,
        figratio=(10, 6)
    )

    if not isinstance(axlist, list):
        ax = axlist
    else:
        ax = axlist[0]

    if not is_single_ticker:

        low_y = plot_data.min().min * 0.95
        high_y = plot_data.max().max() * 1.05
        ax.set_ylim(low_y, high_y)
        ax.set_ylabel('Percent Change from Start (%)')

    if not is_single_ticker:
        ax.legend([ticker for ticker in ticker_list])

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

