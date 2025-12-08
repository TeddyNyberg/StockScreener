
open_detail_windows = []

def open_detail_window(result):
    from app.ui.detail_window import DetailsWindow
    new_details_window = DetailsWindow(result[0], result[2], result[1])
    open_detail_windows.append(new_details_window)
    new_details_window.show()


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
