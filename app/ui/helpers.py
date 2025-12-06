from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy

from app.data.yfinance_fetcher import get_close_on

from app.search.ticker_lookup import lookup_tickers
from app.ml_logic.tester import continue_backtest
from app.db.db_handler import DB, init_db
from app.data.data_cache import get_yfdata_cache
import pandas as pd
from app.ml_logic.strategy import calculate_kelly_allocations
from app.data.ticker_source import get_sp500_tickers
import time
import torch

# just window stuff how it looks, buttons, etc.

open_detail_windows = []

def start_application():
    from app.ui.main_window import MainWindow
    print("Starting app...")
    #print_model_characteristics("NYBERG-A")

    #calculate_kelly_allocations("A",False)

    run_backtesting()
    with DB() as conn:
        init_db(conn)
    window = MainWindow()
    window.show()

# result is structured as name_and_price, chart, data
def open_window_from_ticker(result):
    from app.ui.detail_window import DetailsWindow
    new_details_window = DetailsWindow(result[0], result[2], result[1])
    open_detail_windows.append(new_details_window)
    new_details_window.show()



def clear_layout(layout):
    while layout.count() > 0:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()

def open_watchlist(self):
    from app.ui.watchlist_window import WatchlistWindow
    if self.watch_window is None:
        self.watch_window = WatchlistWindow()
    self.watch_window.show()

def open_portfolio(self):
    from app.ui.portfolio_window import PortfolioWindow
    if self.portfolio_window is None:
        self.portfolio_window = PortfolioWindow()
    self.portfolio_window.show()


def make_buttons(button_map, layout):
    button_layout = QHBoxLayout()
    for label, (func, get_args_func) in button_map.items():
        btn = QPushButton(label)
        btn.clicked.connect(lambda _, f=func, a_func=get_args_func: f(*a_func()))
        button_layout.addWidget(btn)
    spacer = QSpacerItem(40, 20, QSizePolicy.Expanding)
    button_layout.addItem(spacer)
    layout.addLayout(button_layout)

def lookup_and_open_details(ticker, display_error_func=None):
    if not ticker:
        if display_error_func:
            display_error_func("Please enter a ticker symbol.")
        return
    result = lookup_tickers(ticker)
    if not result:
        if display_error_func:
            display_error_func(f"Could not find data for ticker: {ticker}.")
        return
    open_window_from_ticker(result)





#TODO: this may go into search helpers....
def print_model_characteristics(nyberg_ticker):

    df_n, df_spy = get_yfdata_cache([nyberg_ticker, "SPY"],"1Y")
    n_data = df_n["Close"]
    spy = df_spy["Close"]

    combined_data = pd.DataFrame({
        'Nyberg_Close': n_data,
        'SPY_Close': spy
    }).dropna()

    combined_data['Nyberg_Return'] = combined_data['Nyberg_Close'].pct_change().mul(100)
    combined_data['SPY_Return'] = combined_data['SPY_Close'].pct_change().mul(100)

    combined_data.dropna(inplace=True)

    nyberg_better = combined_data['Nyberg_Return'] > combined_data['SPY_Return']

    nyberg_wins = nyberg_better.sum()
    spy_wins = (~nyberg_better).sum()

    spy_on_nyberg_win_returns = combined_data.loc[nyberg_better, 'SPY_Return']

    avg_spy_return_on_nyberg_win = spy_on_nyberg_win_returns.mean()

    spy_on_nyberg_win_and_spy_up = spy_on_nyberg_win_returns[spy_on_nyberg_win_returns > 0].mean()
    spy_on_nyberg_win_and_spy_down = spy_on_nyberg_win_returns[spy_on_nyberg_win_returns <= 0].mean()

    avg_abs_nyberg_move = combined_data['Nyberg_Return'].abs().mean()
    avg_abs_spy_move = combined_data['SPY_Return'].abs().mean()

    print("\n**Volatility Comparison (Average Absolute Daily % Move):**")
    print(f"- **{nyberg_ticker} Average Absolute % Move:** {avg_abs_nyberg_move:.4f}%")
    print(f"- **SPY Average Absolute % Move:** {avg_abs_spy_move:.4f}%")

    print(f"**Performance Metrics: {nyberg_ticker} vs. SPY**")
    print("---")

    print(f"**Total Trading Days Analyzed:** {len(combined_data)}")

    print("\n**Comparison Counts (Daily Returns):**")
    print(f"- **{nyberg_ticker} does better:** {nyberg_wins} days")
    print(f"- **SPY does better (or ties):** {spy_wins} days")

    print("\n**Conditional SPY Return (when Nyberg does better):**")
    print(f"- **Average SPY % Change:** {avg_spy_return_on_nyberg_win:.4f}%")

    print("\n**SPY's Conditional Performance Breakdown (on Nyberg's better days):**")
    print(f"- **SPY's Avg % Change on UP Days:** {spy_on_nyberg_win_and_spy_up:.4f}%")
    print(f"- **SPY's Avg % Change on DOWN Days:** {spy_on_nyberg_win_and_spy_down:.4f}%")

def run_backtesting():
    continue_backtest(version="A")
    continue_backtest(version="B", tuning_period="weekly")
    continue_backtest(version="C")
    continue_backtest(version="D", only_largest=True)