from PySide6.QtWidgets import QHBoxLayout, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QSpacerItem, QSizePolicy
from PySide6.QtCore import Signal, QObject
from .search import lookup_ticker


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Screener")

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
        self.result_label.setText(result)

class DetailsWindow(QWidget):
    pass