
from datetime import datetime
from db import get_watchlist, add_watchlist, rm_watchlist

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout,
                               QLineEdit, QPushButton, QSpacerItem, QTableWidget, QTableWidgetItem,
                               QSizePolicy, QGridLayout, QMenu)
from app.search import (lookup_tickers, get_chart, get_financial_metrics, get_balancesheet, get_info, get_date_range,
                        get_price, get_close_on, get_open_on)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

from data import (fetch_stock_data)

from ml_logic import predict_single_ticker, calculate_kelly_allocations
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
        make_buttons({"Model": (self.handle_model, lambda: [])}, top_layout)

        spacer = QSpacerItem(400, 20, QSizePolicy.Expanding)
        top_layout.addItem(spacer)

        btn = QPushButton("Watchlist")
        btn.clicked.connect(lambda _: open_watchlist(self))
        top_layout.addWidget(btn)

        self.search_widget = SearchWidget()
        self.search_widget.search_requested.connect(open_window_from_ticker)
        self.search_widget.message_displayed.connect(self.update_status_message)
        top_layout.addWidget(self.search_widget)

        main_layout.addLayout(top_layout)

        self.result_label = QLabel("Enter a ticker and press Enter")
        main_layout.addWidget(self.result_label)

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
        self.setWindowTitle(f"Details for {name_and_price[0]["ticker"]}")
        self.ticker_data = name_and_price

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout

        top_row_layout = QHBoxLayout()
        top_row_layout.addWidget(QLabel(f"Name: {name_and_price[0]["name"]}"))

        btn = QPushButton("Add to Watchlist")
        btn.clicked.connect(lambda _: add_watchlist(name_and_price[0]["ticker"]))
        top_row_layout.addWidget(btn)

        top_row_layout.addItem(QSpacerItem(40, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        btn = QPushButton("Watchlist")
        btn.clicked.connect(lambda _: open_watchlist(self))
        top_row_layout.addWidget(btn)

        self.search_widget = SearchWidget()
        self.search_widget.search_requested.connect(open_window_from_ticker)
        self.search_widget.message_displayed.connect(self.update_status_message)
        top_row_layout.addWidget(self.search_widget)

        layout.addLayout(top_row_layout)

        second_row_layout = QHBoxLayout()
        second_row_layout.addWidget(QLabel(f"Price: {name_and_price[0]["price"]} {name_and_price[0]["currency"]}"))
        second_row_layout.addItem(QSpacerItem(40, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.compare_button = QPushButton("Compare")
        self.compare_button.clicked.connect(self.compare)
        second_row_layout.addWidget(self.compare_button)

        layout.addLayout(second_row_layout)

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

        back_test_button = QPushButton("Back Test")
        back_test_button.clicked.connect(lambda: handle_backtest())
        top_row_layout.addWidget(back_test_button)


        layout.addLayout(top_row_layout)

        # TODO: add button to get data maybe?

        middle_layout = QHBoxLayout()

        train_button = QPushButton("Train on Cloud")
        train_button.clicked.connect(self.train_on_cloud)
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

            current_price = fetch_stock_data(ticker, "1M", "1D")["Close"].iloc[-1]
            message = f"Prediction for {ticker}: {prediction:.2f} (Current: {current_price:.2f})"
            self.update_status_message(message)

            start, end = get_date_range("1M")
            df = fetch_stock_data(ticker, start, end)
            print(f"most recent day {df["Close"].iloc[-1]}")
            print(f"Predicted next day value: {prediction}")

        except Exception as e:
            self.update_status_message(f"Error predicting for {ticker}: {e}")
            print(f"Prediction error: {e}")


    def train_on_cloud(self):
        try:
            subprocess.Popen([sys.executable, os.path.join('app', 'sage.py')], cwd=os.getcwd())
            print("SageMaker training job started. Check your AWS console for progress.")
        except FileNotFoundError:
            print("Error: sage.py not found. Make sure the file exists.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def show_kelly_bet(self, start=None, end=None):
        final_allocations = calculate_kelly_allocations("6M")

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
        goog = fetch_stock_data("GOOG", start, end)
        print(goog.tail(1))



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


def handle_backtest(start_date_str: str = "2025-10-04", initial_capital: float = 100000.0, lookback_period: str = "6M"):

    print(f"Starting backtest from {start_date_str} with ${initial_capital:,.2f}...")

    daily_position_reports = {}
    daily_pnl_reports = {}

    try:
        start_date = pd.to_datetime(start_date_str)
        today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d')) - pd.Timedelta(days=1)

        if start_date > today:
            print("Start date is in the future or today. Cannot run backtest.")
            return

        date_range = pd.date_range(start_date, today, freq='B')  # Business days

        portfolio_df = pd.DataFrame(index=date_range, columns=['Total_Value', 'Cash'], dtype=float)
        portfolio_df.loc[date_range[0], "Total_Value"] = initial_capital
        portfolio_df.loc[date_range[0], "Cash"] = initial_capital

        current_holdings = {}  # ticker: {"shares": float, "entryprice":float}

        for i in range(1, len(date_range)):
            current_day = date_range[i]
            prev_day = date_range[i - 1]

            print(current_day)
            print(prev_day)

            prev_day_cash_raw = portfolio_df.loc[prev_day, 'Cash']
            if pd.isna(prev_day_cash_raw):
                day_start_capital = 0.0
                print(f"Error: Cash for {prev_day.strftime('%Y-%m-%d')} is NaN. Assuming 0 capital.")
            else:
                day_start_capital = float(prev_day_cash_raw)

            # 0.0005 5bp
            # 0 for now
            transaction_cost_pct = 0
            daily_pnl_list = []

            tickers_to_remove = list(current_holdings.keys())
            for ticker in tickers_to_remove: #LIQUIDATE
                holding = current_holdings.get(ticker, {})
                shares = holding.get("shares", 0)
                entry_price = holding.get("entry_price", None)

                if shares == 0:
                   continue

                closing_price = get_close_on(ticker, prev_day)  # Price from the previous day's close

                if closing_price is not None and entry_price is not None:
                    sale_value = shares * closing_price
                    transaction_cost = sale_value * transaction_cost_pct

                    purchase_value = shares * entry_price
                    gain_loss = sale_value - purchase_value
                    day_start_capital += sale_value - transaction_cost

                    daily_pnl_list.append({
                        "Ticker": ticker,
                        "Shares_Sold": shares,
                        "Entry_Price": entry_price,
                        "Exit_Price": closing_price,
                        "Sale_Value": sale_value,
                        "Transaction_Cost": transaction_cost,
                        "Gain_Loss": gain_loss
                    })
                else:
                    print(
                        f"Warning: Price unavailable for {ticker} on {prev_day}. Assuming zero contribution to daily capital.")

            if daily_pnl_list:
                daily_pnl_reports[current_day.strftime('%Y-%m-%d')] = pd.DataFrame(daily_pnl_list)

            portfolio_df.loc[current_day, 'Total_Value'] = day_start_capital

            data_end_date = prev_day

            kelly_allocations = calculate_kelly_allocations(
                lookback_period,
                data_end_date
            )

            total_capital_available = day_start_capital
            cash_for_trading = day_start_capital  # We start with cash = total capital, then sell/buy

            new_holdings = {}


            daily_allocations_list = []

            total_allocation_value = 0

            for ticker, kelly_fraction, _ in kelly_allocations:
                target_value = total_capital_available * kelly_fraction
                close_price = get_close_on(ticker, prev_day)

                if close_price is not None:
                    buy_value = min(target_value, cash_for_trading)

                    shares_to_buy = buy_value / close_price
                    transaction_cost = buy_value * transaction_cost_pct

                    if cash_for_trading >= (buy_value + transaction_cost):
                        cash_for_trading -= (buy_value + transaction_cost)
                        new_holdings[ticker] = {"shares": shares_to_buy, "entry_price": close_price, "closing_price": close_price}

                        total_allocation_value += buy_value

                        daily_allocations_list.append({
                            "Ticker": ticker,
                            "Allocation_Percent": kelly_fraction,
                            "Entry_Price": close_price,
                            "Target_Value": target_value,
                            "Actual_Shares": shares_to_buy
                        })

            current_holdings = new_holdings
            portfolio_df.loc[current_day, 'Cash'] = float(cash_for_trading)
            portfolio_df.loc[current_day, 'Total_Value'] = float(cash_for_trading + total_allocation_value)

            if daily_allocations_list:
                allocations_df = pd.DataFrame(daily_allocations_list)
                daily_position_reports[current_day.strftime("%Y-%m-%d")] = allocations_df

            # print(f"Day {current_day.strftime('%Y-%m-%d')} | End Value: ${portfolio_df.loc[current_day, 'Total_Value']:,.2f} | Cash: ${cash_for_trading:,.2f}")


        file_path = 'backtest_results.xlsx'
        writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
        portfolio_df.to_excel(writer, sheet_name="Summary_Performance")

        #for pnl sheet, to see loss and gain on 5/1, look at sheet named 5/2 pnl
        #for allocations, the stocks syou see on 5/1 are the stocks you own on 5/1

        for date_str, df in daily_position_reports.items():
            sheet_name = f"{date_str}_Allocations"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        for date_str, df in daily_pnl_reports.items():
            sheet_name = f"{date_str}_PnL"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.close()
        print("\n" + "=" * 50)
        print(f"Backtest complete. Final Value: ${portfolio_df.iloc[-1]['Total_Value']:,.2f}")
        print(f"Results saved to {file_path}")
        print("=" * 50)

    except Exception as e:
        print(f"Backtesting error: {e}")

    print(portfolio_df)

    return portfolio_df
