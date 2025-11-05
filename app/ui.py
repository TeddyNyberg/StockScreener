from app.db.db_handler import get_watchlist, add_watchlist, rm_watchlist, get_portfolio, buy_stock, sell_stock
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout,
                               QLineEdit, QPushButton, QSpacerItem, QTableWidget, QTableWidgetItem,
                               QSizePolicy, QGridLayout, QMenu, QSpinBox)
from app.search.charting import get_chart
from app.search.ticker_lookup import lookup_tickers
from app.utils import get_date_range
from app.data.yfinance_fetcher import get_info, get_financial_metrics, get_balancesheet, get_price
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from app.data.yfinance_fetcher import get_historical_data
from app.ml_logic.strategy import calculate_kelly_allocations, predict_single_ticker
from app.ml_logic.tester import handle_backtest, continue_backtest
import subprocess
import sys
from settings import *



# just window stuff how it looks, buttons, etc.

open_detail_windows = []


# result is structured as name_and_price, chart, data
def open_window_from_ticker(result):
    new_details_window = DetailsWindow(result[0], result[2], result[1])
    open_detail_windows.append(new_details_window)
    new_details_window.show()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()


        self.setWindowTitle("Stock Screener")
        main_layout = QVBoxLayout()

        self.resize(800, 600)

        top_layout = QHBoxLayout()
        self.model_window = None
        self.watch_window = None
        self.portfolio_window = None

        m_btn = QPushButton("Model")
        m_btn.clicked.connect(self.handle_model)
        top_layout.addWidget(m_btn, alignment=Qt.AlignmentFlag.AlignTop)

        spacer = QSpacerItem(400, 20, QSizePolicy.Expanding)
        top_layout.addItem(spacer)

        watchlist_port_layout = QVBoxLayout()


        w_btn = QPushButton("Watchlist")
        w_btn.clicked.connect(lambda _: open_watchlist(self))
        watchlist_port_layout.addWidget(w_btn)

        i_btn = QPushButton("Investments")
        i_btn.clicked.connect(lambda _: open_portfolio(self))
        watchlist_port_layout.addWidget(i_btn)


        top_layout.addLayout(watchlist_port_layout)

        self.search_widget = SearchWidget()
        self.search_widget.search_requested.connect(open_window_from_ticker)
        self.search_widget.message_displayed.connect(self.update_status_message)
        top_layout.addWidget(self.search_widget, alignment=Qt.AlignmentFlag.AlignTop)

        main_layout.addLayout(top_layout)

        spy, spy_chart, data = lookup_tickers("^SPX")
        self.canvas = CustomChartCanvas(data, spy_chart)

        # this is a chart, which takes data and chart ------------|
        # it calls custchart                                      |
        # cust chart takes data and figure                        |
        # cust chart shows it and adds hoverability               |
        # we get data and chart and spy from lookup_tickers <-----|
        # lookup returns a [], chart, and data
        # lookup gets chart and data from get_chart
        # get_chart figures and plot_data df

        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def update_status_message(self, message):
        self.result_label = QLabel(message)

    def handle_model(self):
        if self.model_window is None:
            self.model_window = ModelWindow()
        self.model_window.show()

def clear_layout(layout):
    while layout.count() > 0:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.deleteLater()

def open_watchlist(self):
    if self.watch_window is None:
        self.watch_window = WatchlistWindow()
    self.watch_window.show()

def open_portfolio(self):
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


class DetailsWindow(QMainWindow):
    def __init__(self, name_and_price, pricing_data, chart):
        super().__init__()

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
        btn.clicked.connect(lambda _: add_watchlist(name_and_price[0]["ticker"]))
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
        btn = QPushButton("Watchlist")
        btn.clicked.connect(lambda _: open_watchlist(self))
        wl_sw_layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignRight)
        self.search_widget = SearchWidget()
        self.search_widget.search_requested.connect(open_window_from_ticker)
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
            df = pd.DataFrame(info.items(), columns=["Metric", "Value"])
            table_widget = QTableWidget()
            table_widget.setRowCount(df.shape[0])
            table_widget.setColumnCount(df.shape[1])
            for row_index in range(df.shape[0]):
                for col_index in range(df.shape[1]):
                    value = df.iloc[row_index, col_index]
                    table_widget.setItem(row_index, col_index, QTableWidgetItem(str(value)))
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

            df = pd.DataFrame(info.items(), columns=["Metric", "Value"])

            table_widget = QTableWidget()
            table_widget.setRowCount(df.shape[0])
            table_widget.setColumnCount(df.shape[1])
            for row_index in range(df.shape[0]):
                for col_index in range(df.shape[1]):
                    value = df.iloc[row_index, col_index]
                    table_widget.setItem(row_index, col_index, QTableWidgetItem(str(value)))
            self.content_layout.addWidget(table_widget)
            return

        else:
            pass
        if not df.empty:
            table_widget = QTableWidget()

            table_widget.setRowCount(df.shape[0])
            table_widget.setColumnCount(df.shape[1])

            table_widget.setHorizontalHeaderLabels([str(col.date()) for col in df.columns])
            table_widget.setVerticalHeaderLabels(df.index)

            for row_index in range(df.shape[0]):
                for col_index in range(df.shape[1]):
                    value = df.iloc[row_index, col_index]
                    table_widget.setItem(row_index, col_index, QTableWidgetItem(str(value)))
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
        if self.trading_window is None:
            self.trading_window = TradingWindow(ticker, price)
        self.trading_window.show()




# chart figure is a list of charts??????
# TODO: make it accept possible a list of pricing data, so it works for comparison too fml
class CustomChartCanvas(FigureCanvas):
    def __init__(self, chart_data, chart_figure, parent=None):
        self.figure = chart_figure
        self.chart_data = chart_data

        self.axes = self.figure.get_axes()[0]
        super().__init__(self.figure)

        self.setParent(parent)
        self.mpl_connect('motion_notify_event', self.on_hover)

        # for pointer
        self.v_line = self.axes.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        self.h_line = self.axes.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        self.v_line.set_visible(False)
        self.h_line.set_visible(False)

        self.annot = self.axes.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
                                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        self.annot.set_visible(False)

    def on_hover(self, event):
        if event.inaxes != self.axes:
            self.v_line.set_visible(False)
            self.h_line.set_visible(False)
            self.annot.set_visible(False)
            self.draw_idle()
            return

        x, y = event.xdata, event.ydata

        if x is not None and y is not None:
            x_int = int(round(x))
            data = self.chart_data

            if 0 <= x_int < len(data.index):
                date = data.index[x_int].strftime('%Y-%m-%d')
                price = data.iloc[x_int]['Close']

                self.v_line.set_xdata([x, x])
                self.h_line.set_ydata([price, price])

                self.annot.xy = (x, y)
                self.annot.set_text(f"Date: {date}\nPrice: {price:.2f}")
                self.annot.set_visible(True)
                self.v_line.set_visible(True)
                self.h_line.set_visible(True)

                self.draw_idle()


class SearchWidget(QWidget):
    search_requested = Signal(object)
    message_displayed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setMaximumWidth(250)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.search_bar_input = QLineEdit()
        self.search_bar_input.setPlaceholderText("Enter ticker")
        self.search_bar_input.returnPressed.connect(self.handle_search)
        layout.addWidget(self.search_bar_input)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.handle_search)
        layout.addWidget(self.search_button)

        self.setLayout(layout)

    def handle_search(self):
        ticker = self.search_bar_input.text().strip().upper()
        lookup_and_open_details(ticker, display_error_func=self.message_displayed.emit)
        self.search_bar_input.clear()



class ModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout

        top_row_layout = QHBoxLayout()
        top_row_layout.addItem(QSpacerItem(90, 0))

        next_day_picks = QPushButton("Next Day Picks")
        next_day_picks.clicked.connect(self.show_kelly_bet)
        top_row_layout.addWidget(next_day_picks)


        #TODO: make this filename universal?
        back_test_button = QPushButton("Back Test")
        back_test_button.clicked.connect(lambda: continue_backtest("backtest_results_jan.xlsx", "Summary_Performance"))
        top_row_layout.addWidget(back_test_button)


        layout.addLayout(top_row_layout)

        # TODO: add button to get data maybe?

        middle_layout = QHBoxLayout()

        train_button = QPushButton("Train on Cloud")
        train_button.clicked.connect(lambda: train_on_cloud())
        middle_layout.addWidget(train_button)

        self.search_bar_input = QLineEdit()
        self.search_bar_input.setPlaceholderText("Enter ticker")
        self.next_day_button = QPushButton("Next Day")
        self.next_day_button.clicked.connect(self.get_next_day)
        middle_layout.addWidget(self.next_day_button)
        middle_layout.addWidget(self.search_bar_input)
        layout.addLayout(middle_layout)

        self.third_layout = QGridLayout()

        self.setCentralWidget(central_widget)

    def update_status_message(self, message):
        self.result_label = QLabel(message)


    def get_next_day(self):
        ticker = self.search_bar_input.text().strip().upper()
        if not ticker:
            self.update_status_message("Please enter a ticker symbol.")
            return
        try:
            prediction = predict_single_ticker(ticker)

            start, end = get_date_range("1M")
            df = get_historical_data(ticker, start, end)
            print(f"most recent day {df['Close'].iloc[-1]}")
            print(f"Predicted next day value: {prediction}")

        except Exception as e:
            self.update_status_message(f"Error predicting for {ticker}: {e}")
            print(f"Prediction error: {e}")

    def show_kelly_bet(self):
        final_allocations = calculate_kelly_allocations()

        self.third_layout.addWidget(QLabel("Ticker"), 0, 0)
        self.third_layout.addWidget(QLabel("Allocation"), 0, 1)
        self.third_layout.addWidget(QLabel("Proj Return"), 0, 2)

        i = 1
        for ticker, normalized_allocation, mu in final_allocations:
            allocation_text = f"{normalized_allocation * 100:.2f}%"
            mu_text = f"{(mu * 100):+.4f}%"
            self.third_layout.addWidget(QLabel(ticker), i, 0)
            self.third_layout.addWidget(QLabel(allocation_text), i, 1)
            self.third_layout.addWidget(QLabel(mu_text), i, 2)
            i += 1

        self.layout.addLayout(self.third_layout)

        print(f"\nFinal Total Portfolio Allocation: {sum(a for _, a, _ in final_allocations) * 100:.2f}%")

        start, end = get_date_range("6M")
        goog = get_historical_data("GOOG", start, end) # sanity check, shows latest data available for model
        print(goog.tail(1))

def train_on_cloud():
    try:
        subprocess.Popen([sys.executable, os.path.join('app', 'sage.py')], cwd=os.getcwd())
        print("SageMaker training job started. Check your AWS console for progress.")
    except FileNotFoundError:
        print("Error: sage.py not found. Make sure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

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
