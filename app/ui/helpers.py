from PySide6.QtWidgets import QHBoxLayout, QPushButton, QSpacerItem, QSizePolicy
from app.search.ticker_lookup import lookup_tickers
from app.ml_logic.tester import handle_backtest, continue_backtest
from app.db.db_handler import DB, init_db
import threading

# just window stuff how it looks, buttons, etc.

open_detail_windows = []

def start_application():
    from app.ui.main_window import MainWindow
    print("Starting app...")
    backtest_thread = threading.Thread(target=continue_backtest, args=["backtest_results_jan.csv"])
    backtest_thread.daemon = True
    backtest_thread.start()
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


