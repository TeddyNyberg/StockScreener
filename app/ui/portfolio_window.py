from app.db.db_handler import get_portfolio, buy_stock, sell_stock
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QGridLayout,
                               QSpinBox)

from app.data.yfinance_fetcher import get_price
from app.ui.helpers import clear_layout
from app.ui.watchlist_window import TickerButton


class PortfolioWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Investments")

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

        portfolio = get_portfolio()

        new_list_layout = QGridLayout()

        new_list_layout.addWidget(QLabel("Ticker"), 0, 0)
        new_list_layout.addWidget(QLabel("Price"), 0, 1)
        new_list_layout.addWidget(QLabel("Shares Owned"), 0, 2)
        new_list_layout.addWidget(QLabel("Cost Basis"), 0, 3)
        new_list_layout.addWidget(QLabel("Average Cost Basis"), 0, 4)
        new_list_layout.addWidget(QLabel("Total Gain"), 0, 5)
        new_list_layout.addWidget(QLabel("Percent Return"), 0, 6)


        i = 1
        for entry in portfolio:
            # entry = (ticker, shares owned, total cost basis)
            ticker = entry[0]
            shares = entry[1]
            total_cost_basis = entry[2]
            avg_cost_basis = total_cost_basis / shares
            cur_price = get_price(ticker)
            profit_loss = (cur_price - float(avg_cost_basis)) * shares
            ticker_button = TickerButton(ticker)
            new_list_layout.addWidget(ticker_button, i, 0)
            new_list_layout.addWidget(QLabel(str(cur_price)), i, 1)
            new_list_layout.addWidget(QLabel(str(shares)), i, 2)
            new_list_layout.addWidget(QLabel(str(total_cost_basis)), i, 3)
            new_list_layout.addWidget(QLabel(str(avg_cost_basis)), i, 4)
            new_list_layout.addWidget(QLabel(str(profit_loss)), i, 5)


            i += 1

        self.list_layout = new_list_layout
        self.layout.addLayout(self.list_layout)


class TradingWindow(QMainWindow):
    def __init__(self, ticker, price):
        super().__init__()

        self.ticker = ticker
        self.price = price
        self.setWindowTitle(f"Trade: {self.ticker}")

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"<b>Stock:</b> {self.ticker}"))
        info_layout.addWidget(QLabel(f"<b>Current Price:</b> ${self.price:.2f}"))
        main_layout.addLayout(info_layout)

        main_layout.addWidget(QLabel("---"))

        quantity_layout = QHBoxLayout()
        quantity_layout.addWidget(QLabel("<b>Quantity:</b>"))

        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(1, 1000000)  # Set a reasonable range
        self.quantity_input.setValue(1)
        self.quantity_input.setToolTip("Number of shares to buy or sell")
        quantity_layout.addWidget(self.quantity_input)

        main_layout.addLayout(quantity_layout)

        trade_buttons_layout = QHBoxLayout()

        self.buy_button = QPushButton("BUY")
        self.buy_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.buy_button.clicked.connect(lambda _: buy_stock(ticker, self.quantity_input.value(), price))
        trade_buttons_layout.addWidget(self.buy_button)

        self.sell_button = QPushButton("SELL")
        self.sell_button.setStyleSheet("background-color: #F44336; color: white;")
        self.sell_button.clicked.connect(lambda: sell_stock(ticker, self.quantity_input.value(), price))
        trade_buttons_layout.addWidget(self.sell_button)

        main_layout.addLayout(trade_buttons_layout)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()
        self.setCentralWidget(central_widget)
        self.resize(300, 150)