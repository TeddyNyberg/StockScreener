from PyQt5.QtWidgets import QMessageBox

from app.db.db_handler import get_watchlist, rm_watchlist
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QMainWindow, QWidget, QLabel, QVBoxLayout, QPushButton, QGridLayout, QMenu, QTableView,
                               QHeaderView)
from app.data.yfinance_fetcher import get_price
from app.ui.models.watchlist_model import WatchlistModel
from app.ui.search_handler import lookup_and_open_details
from app.ui.ui_utils import clear_layout


class WatchlistWindow(QMainWindow):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("Watchlist")
        self.resize(400, 500)

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.verticalHeader().setVisible(False)

        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)

        self.table_view.doubleClicked.connect(self.on_row_double_clicked)

        self.layout.addWidget(self.table_view)
        self.rebuild_display()

    def rebuild_display(self):
        watchlist_data = get_watchlist(self.user_id)
        self.model = WatchlistModel(watchlist_data)
        self.table_view.setModel(self.model)

    def on_row_double_clicked(self, index):
        ticker = self.model.get_ticker(index.row())
        lookup_and_open_details(ticker)

    def show_context_menu(self, pos):
        index = self.table_view.indexAt(pos)
        if not index.isValid():
            return

        ticker = self.model.get_ticker(index.row())
        menu = QMenu(self.table_view)

        rm_action = menu.addAction(f"Remove {ticker}")
        action = menu.exec(self.table_view.viewport().mapToGlobal(pos))
        if action == rm_action:
            self.remove_ticker(ticker)

    def remove_ticker(self, ticker):
        success = rm_watchlist(ticker, self.user_id)
        if success:
            self.rebuild_display()
        else:
            QMessageBox.warning(self, "Error", "Could not remove ticker.")

class TickerButton(QPushButton):
    remove_requested = Signal(str)
    def __init__(self, ticker):
        super().__init__(ticker)
        self.ticker = ticker
        self.clicked.connect(lambda: lookup_and_open_details(self.ticker))


