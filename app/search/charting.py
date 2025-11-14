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


def generate_scatter_plot(nyberg_ticker):

    df_nyberg, df_spy = get_yfdata_cache([nyberg_ticker, "SPY"], "1Y")

    nyberg_returns = df_nyberg['Close'].pct_change().mul(100)
    spy_returns = df_spy['Close'].pct_change().mul(100)

    returns_df = pd.DataFrame({
        'Nyberg_Return': nyberg_returns,
        'SPY_Return': spy_returns
    }).dropna()

    max_abs_return = returns_df[['Nyberg_Return', 'SPY_Return']].abs().max().max() * 1.05
    plot_limit = max(0.1, max_abs_return)

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 8))

    # Scatter plot: Nyberg (X) vs. SPY (Y)
    plt.scatter(
        returns_df['Nyberg_Return'],
        returns_df['SPY_Return'],
        s=70,
        color='#1f77b4',
        alpha=0.6,
        edgecolors='white',
        linewidths=0.8
    )

    # 45-degree line (Y = X) where both returns are equal
    plt.plot([-plot_limit, plot_limit], [-plot_limit, plot_limit],
             'r--',
             alpha=0.5,
             label='Y = X (Equal Returns)')

    plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.8)

    plt.title(f'Volatility and Correlation: {nyberg_ticker} vs. SPY Daily % Returns', fontsize=16, fontweight='bold')
    plt.xlabel(f'{nyberg_ticker} Daily Return (%)', fontsize=14)
    plt.ylabel('SPY Daily Return (%)', fontsize=14)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(-plot_limit, plot_limit)
    plt.ylim(-plot_limit, plot_limit)

    plt.legend(loc='lower right')

    print(f"Generating scatterplot for {nyberg_ticker} vs SPY...")
    plt.show()