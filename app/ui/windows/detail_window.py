from app.db.db_handler import add_watchlist
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QMessageBox)
from app.search.charting import get_chart
from app.search.ticker_lookup import lookup_tickers
from app.data.yfinance_fetcher import get_info, get_financial_metrics, get_balancesheet
from app.ui.chart_canvas import CustomChartCanvas
from app.ui.window_manager import open_detail_window, open_watchlist
from app.ui.ui_utils import clear_layout, make_buttons, create_table_widget_from_data
from app.ui.windows.portfolio_window import TradingWindow
from app.ui.widgets import SearchWidget


class DetailsWindow(QMainWindow):
    def __init__(self, name_and_price, pricing_data, chart, user_id=None):
        super().__init__()
        self.user_id = user_id
        self.comp_ticker = None
        self.watch_window = None
        self.trading_window = None
        self.setWindowTitle(f"Details for {name_and_price[0]['ticker']}")
        self.ticker_data = name_and_price

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout

        nm_wl_layout = QHBoxLayout()
        nm_wl_layout.addWidget(QLabel(f"Name: {name_and_price[0]['name']}"))
        btn = QPushButton("Add to Watchlist")
        btn.clicked.connect(self.handle_add_watchlist)
        nm_wl_layout.addWidget(btn)

        pr_buy_layout = QHBoxLayout()
        pr_buy_layout.addWidget(QLabel(f"Price: {name_and_price[0]['price']} {name_and_price[0]['currency']}"))
        btn = QPushButton("Buy/Sell")
        btn.clicked.connect(lambda _: self.handle_investment_window(name_and_price[0]["ticker"], name_and_price[0]["price"]))
        pr_buy_layout.addWidget(btn)

        left_side = QVBoxLayout()
        left_side.addLayout(nm_wl_layout)
        left_side.addLayout(pr_buy_layout)

        top_row_layout = QHBoxLayout()
        top_row_layout.addLayout(left_side)

        wl_sw_layout = QHBoxLayout()
        btn_wl = QPushButton("Watchlist")
        btn_wl.clicked.connect(self.open_watchlist_check)
        wl_sw_layout.addWidget(btn_wl, alignment=Qt.AlignmentFlag.AlignRight)

        self.search_widget = SearchWidget()
        self.search_widget.search_requested.connect(open_detail_window)
        self.search_widget.message_displayed.connect(self.update_status_message)
        wl_sw_layout.addWidget(self.search_widget)
        wl_sw_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        comp_layout = QHBoxLayout()
        self.compare_button = QPushButton("Compare")
        self.compare_button.clicked.connect(self.compare)
        comp_layout.addWidget(self.compare_button)
        comp_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        right_side = QVBoxLayout()
        right_side.addLayout(wl_sw_layout)
        right_side.addLayout(comp_layout)

        top_row_layout.addLayout(right_side)

        layout.addLayout(top_row_layout)

        self.canvas = CustomChartCanvas(pricing_data, chart)
        self.canvas.setMaximumSize(900, 600)
        layout.addWidget(self.canvas)

        timeframes = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"]
        time_buttons = {
            t: (self.update_chart, lambda t=t: [t, [self.ticker_data[0]["ticker"], self.comp_ticker]])
            for t in timeframes
        }

        info_buttons = {
            "Information": (self.show_fin_info, lambda: ["info", self.ticker_data[0]["ticker"]]),
            "Financials": (self.show_fin_info, lambda: ["financials", self.ticker_data[0]["ticker"]]),
            "Balance Sheet": (self.show_fin_info, lambda: ["balance_sheet", self.ticker_data[0]["ticker"]]),
            "What I care About": (self.show_fin_info, lambda: ["my_chart", self.ticker_data[0]["ticker"]])
        }

        make_buttons(time_buttons, self.layout)
        make_buttons(info_buttons, self.layout)
        self.content_layout = QVBoxLayout()
        self.layout.addLayout(self.content_layout)

        self.setCentralWidget(central_widget)

    def handle_add_watchlist(self):
        if self.user_id is None:
            QMessageBox.warning(self, "Warning", "You need to login first.")
            return
        add_watchlist(self.ticker_data[0]["ticker"], self.user_id)

    def open_watchlist_check(self):
        if self.user_id is None:
            QMessageBox.warning(self, "Warning", "You need to login first.")
            return
        open_watchlist(self)


    def update_chart(self, time, tickers=None):
        if tickers is None:
            new_fig, data = get_chart([self.ticker_data[0]["ticker"]], time)
        else:
            new_fig, data = get_chart(tickers, time)
        ind = self.layout.indexOf(self.canvas)
        self.layout.removeWidget(self.canvas)
        self.canvas.deleteLater()
        self.canvas = CustomChartCanvas(data, new_fig)
        self.canvas.setMaximumSize(900, 600)
        self.layout.insertWidget(ind, self.canvas)

    def show_fin_info(self, metrics, _):
        clear_layout(self.content_layout)
        if metrics == "financials":
            self.content_layout.addWidget(QLabel("--- Financials ---"))
            df = get_financial_metrics(self.ticker_data[0]["ticker"])
        elif metrics == "balance_sheet":
            self.content_layout.addWidget(QLabel("--- Balance Sheet ---"))
            df = get_balancesheet(self.ticker_data[0]["ticker"])
        elif metrics == "info":
            self.content_layout.addWidget(QLabel("--- Info ---"))
            info = get_info(self.ticker_data[0]["ticker"])
            table_widget = create_table_widget_from_data(info=info)
            self.content_layout.addWidget(table_widget)
            return
        elif metrics == "my_chart":
            info = {}
            stock_info_dict = get_info(self.ticker_data[0]["ticker"])
            lookup_stats = ["dividendYield", "beta", "trailingPE", "forwardPE", "volume", "averageVolume", "bid", "ask",
                            "marketCap", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "priceToSalesTrailing12Months",
                            "twoHundredDayAverage", "profitMargins", "heldPercentInsiders", "priceToBook",
                            "earningsQuarterlyGrowth", "debtToEquity", "returnOnEquity", "earningsGrowth",
                            "revenueGrowth", "grossMargins", "trailingPegRatio"]
            for stat in lookup_stats:
                info[stat] = stock_info_dict.get(stat)
            table_widget = create_table_widget_from_data(info=info)
            self.content_layout.addWidget(table_widget)
            return
        else:
            pass
        if not df.empty:
            table_widget = create_table_widget_from_data(df=df)
            self.content_layout.addWidget(table_widget)


    def update_status_message(self, message):
        pass

    def compare(self):
        ticker_to_compare = self.search_widget.search_bar_input.text().strip().upper()

        if not ticker_to_compare:
            self.update_status_message("Please enter a ticker symbol to compare.")
            return

        tickers = [self.ticker_data[0]["ticker"], ticker_to_compare]
        self.comp_ticker = ticker_to_compare

        result = lookup_tickers(tickers)

        if not result:
            self.update_status_message("Could not find data for one or both tickers.")
            return

        self.update_chart("1Y", tickers)

    def handle_investment_window(self, ticker, price):
        if self.user_id is None:
            QMessageBox.warning(self, "Warning", "You need to login first.")
            return
        if self.trading_window is None:
            self.trading_window = TradingWindow(ticker, price, self.user_id)
        self.trading_window.show()
