from app.db.db_handler import get_watchlist, rm_watchlist
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QGridLayout, QMenu
from app.data.yfinance_fetcher import get_price
from app.ui.helpers import clear_layout, lookup_and_open_details


class WatchlistWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Watchlist")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout
        self.setCentralWidget(central_widget)

        self.list_layout = None
        self.rebuild_display()

    def rebuild_display(self):
        if self.list_layout is not None:
            clear_layout(self.list_layout)
            self.layout.removeItem(self.list_layout)
            self.list_layout.deleteLater()

        watchlist = get_watchlist()

        new_list_layout = QGridLayout()

        new_list_layout.addWidget(QLabel("Ticker"), 0, 0)
        new_list_layout.addWidget(QLabel("Price"), 0, 1)

        i = 1
        for entry in watchlist:
            ticker = entry[0]
            ticker_button = TickerButton(ticker)
            ticker_button.remove_requested.connect(self.remove_ticker)
            new_list_layout.addWidget(ticker_button, i, 0)

            price_label_text = str(get_price(ticker))
            new_list_layout.addWidget(QLabel(price_label_text), i, 1)
            i += 1

        self.list_layout = new_list_layout
        self.layout.addLayout(self.list_layout)

    def remove_ticker(self, ticker):
        rm_watchlist(ticker)
        self.rebuild_display()

class TickerButton(QPushButton):
    remove_requested = Signal(str)
    def __init__(self, ticker):
        super().__init__(ticker)
        self.ticker = ticker
        self.clicked.connect(lambda: lookup_and_open_details(self.ticker))

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        action_b = menu.addAction("Remove Ticker")
        action_b.triggered.connect(lambda: self.remove_requested.emit(self.ticker))
        menu.exec(event.globalPos())
