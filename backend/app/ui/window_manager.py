
open_detail_windows = []

CURRENT_USER_ID = None

def open_detail_window(result):
    from backend.app.ui.windows.detail_window import DetailsWindow
    new_details_window = DetailsWindow(result[0], result[2], result[1], CURRENT_USER_ID)
    open_detail_windows.append(new_details_window)
    new_details_window.show()


def open_watchlist(self):
    from backend.app.ui.windows.watchlist_window import WatchlistWindow
    if self.watch_window is None:
        self.watch_window = WatchlistWindow(CURRENT_USER_ID)
    self.watch_window.show()


def open_portfolio(self):
    from backend.app.ui.windows.portfolio_window import PortfolioWindow
    if self.portfolio_window is None:
        self.portfolio_window = PortfolioWindow(CURRENT_USER_ID)
    self.portfolio_window.show()

def set_user_session(user_id):
    global CURRENT_USER_ID
    CURRENT_USER_ID = user_id
    print("UPDATES to: ", CURRENT_USER_ID)