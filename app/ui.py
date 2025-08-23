from PySide6.QtWidgets import QHBoxLayout, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QSpacerItem, QSizePolicy
from PySide6.QtCore import Signal, QObject
from .search import lookup_ticker, get_chart
from PySide6.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


#just window stuff how it looks, buttons, etc.
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Screener")
        self.details_window=None
        main_layout = QVBoxLayout()

        self.resize(800, 600)

        top_layout = QHBoxLayout()

        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        top_layout.addItem(spacer)

        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Enter ticker")
        self.search_bar.returnPressed.connect(self.handle_search)

        self.result_label = QLabel("Enter a ticker and press Enter")

        top_layout.addWidget(self.search_bar)

        main_layout.addLayout(top_layout)

        self.result_label = QLabel("Enter a ticker and press Enter")
        main_layout.addWidget(self.result_label)

        self.setLayout(main_layout)

    def handle_search(self):
        ticker = self.search_bar.text().strip().upper()
        if not ticker:
            self.result_label.setText("Please enter a ticker symbol.")
            return
        result = lookup_ticker(ticker)
        if not result:
            self.result_label.setText("Ticker not Found")
            return
        self.details_window = DetailsWindow(result)
        self.details_window.show()
        self.hide()


class DetailsWindow(QWidget):
    def __init__(self, ticker_data):
        super().__init__()
        self.setWindowTitle(f"Details for {ticker_data["ticker"]}")
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Name: {ticker_data["name"]}"))
        layout.addWidget(QLabel(f"Price: {ticker_data["price"]} {ticker_data["currency"]}"))

        self.canvas = FigureCanvas(ticker_data["chart"])
        layout.addWidget(self.canvas)

        self.setLayout(layout)
