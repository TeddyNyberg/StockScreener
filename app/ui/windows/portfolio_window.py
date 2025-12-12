from app.db.db_handler import get_portfolio, buy_stock, sell_stock, get_balance
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QSpinBox,
                               QTableView, QHeaderView)
from app.ui.models.portfolio_model import PortfolioModel
from app.ui.search_handler import lookup_and_open_details


class PortfolioWindow(QMainWindow):
    def __init__(self, user_id):
        super().__init__()
        self.user_id = user_id
        self.setWindowTitle("Investments")
        self.resize(900, 600)

        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.total_value_label = QLabel("Total Value: $0,00")
        self.total_value_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.total_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.total_value_label)

        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_view.doubleClicked.connect(self.on_row_double_clicked)
        self.layout.addWidget(self.table_view)

        self.rebuild_display()

    def rebuild_display(self):
        portfolio_data = get_portfolio(self.user_id)
        cash_balance = get_balance(self.user_id)
        self.model = PortfolioModel(portfolio_data, cash_balance)
        self.table_view.setModel(self.model)

        self.total_value_label.setText(f"Total Portfolio Value: ${self.model.total_value:,.2f}")

    def on_row_double_clicked(self, index):
        ticker_index = self.model.index(index.row(), 0)
        ticker = self.model.data(ticker_index, role=Qt.DisplayRole)
        if ticker != "CASH":
            lookup_and_open_details(ticker)

class TradingWindow(QMainWindow):
    def __init__(self, ticker, price, user_id):
        super().__init__()

        self.user_id = user_id
        self.ticker = ticker
        self.price = price
        self.setWindowTitle(f"Trade: {self.ticker}")

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"<b>Stock:</b> {self.ticker}"))
        info_layout.addWidget(QLabel(f"<b>Current Price:</b> ${self.price:.2f}"))
        main_layout.addLayout(info_layout)

        self.cash_balance = get_balance(self.user_id)
        cash_label = QLabel(f"Available Cash: ${self.cash_balance:,.2f}")
        cash_label.setStyleSheet("color: gray; font-style: italic;")
        main_layout.addWidget(cash_label)

        main_layout.addWidget(QLabel("---"))

        quantity_layout = QHBoxLayout()
        quantity_layout.addWidget(QLabel("<b>Quantity:</b>"))

        self.quantity_input = QSpinBox()
        self.quantity_input.setRange(1, 1000000)
        self.quantity_input.setValue(1)
        self.quantity_input.setToolTip("Number of shares to buy or sell")
        quantity_layout.addWidget(self.quantity_input)

        main_layout.addLayout(quantity_layout)

        trade_buttons_layout = QHBoxLayout()

        self.buy_button = QPushButton("BUY")
        self.buy_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.buy_button.clicked.connect(lambda _: buy_stock(ticker, self.quantity_input.value(), price, self.user_id))
        trade_buttons_layout.addWidget(self.buy_button)

        self.sell_button = QPushButton("SELL")
        self.sell_button.setStyleSheet("background-color: #F44336; color: white;")
        self.sell_button.clicked.connect(lambda: sell_stock(ticker, self.quantity_input.value(), price, self.user_id))
        trade_buttons_layout.addWidget(self.sell_button)

        main_layout.addLayout(trade_buttons_layout)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)

        main_layout.addStretch()
        self.setCentralWidget(central_widget)
        self.resize(300, 150)