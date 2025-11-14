import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
from app.data.data_cache import get_yfdata_cache
from config import *



# limitation of get_chart: it only works with 1 or 2 tickers.
# also all supporting functions only work with 1 or 2 tickers.
# its all hardcoded
# TODO: allow adding more tickers for comparison
def get_chart(tickers, time):
    is_single_ticker = len(tickers) == 1 or tickers[1] is None

    plot_data, second_data = get_yfdata_cache(tickers, time)

    title = _get_title(tickers)

    if not is_single_ticker:
        len1 = len(plot_data)
        len2 = len(second_data)
        if len1 > len2:
            plot_data = plot_data.iloc[-len2:]
        elif len2 > len1:
            second_data = second_data.iloc[-len1:]
        plot_data, second_data = _normalize(plot_data, second_data)
        print(plot_data)
        print(second_data)

    plot_kwargs = {
        "type": "line",
        "style": DEFAULT_STYLE,
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
        low_y, high_y = _get_y_bounds(plot_data, second_data, is_single_ticker)
        ax.set_ylim(low_y, high_y)

    return fig, plot_data



def _get_title(tickers):
    title = f"{tickers[0]} Stock Price"
    if len(tickers) > 1 and tickers[1] is not None:
        title = f"Price Comparison of {', '.join(tickers)}"
    return title



def _normalize(df1, df2):
    df1 = (df1.div(df1.iloc[0]) - 1) * 100
    df2 = (df2.div(df2.iloc[0]) - 1) * 100
    return df1, df2


def _get_y_bounds(df1, df2, is_single):
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

