import io
import os
import tarfile
import boto3
import numpy
import sklearn

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout,
                               QLineEdit, QPushButton, QSpacerItem, QTableWidget, QTableWidgetItem,
                               QSizePolicy)
from app.search import lookup_tickers, get_chart, get_financial_metrics, get_balancesheet, get_info, get_date_range
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from model import pred_next_day
from modeltosm import get_scaler
from data import feat_engr, df_to_tensor_with_dynamic_ids, fetch_stock_data, DataHandler
import torch
import torch.serialization
import subprocess
import sys

MODEL_ARTIFACTS_PREFIX = 'pytorch-training-2025-09-12-15-43-24-377/source/sourcedir.tar.gz'
S3_BUCKET_NAME = "stock-screener-bucker"
s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

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
        make_buttons({"Model": (self.handle_model, lambda: [])}, top_layout)

        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding)
        top_layout.addItem(spacer)



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
        self.setWindowTitle(f"Details for {name_and_price[0]["ticker"]}")
        self.ticker_data = name_and_price

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout

        top_row_layout = QHBoxLayout()
        top_row_layout.addWidget(QLabel(f"Name: {name_and_price[0]["name"]}"))
        top_row_layout.addItem(QSpacerItem(40, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
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
        if not ticker:
            self.message_displayed.emit("Please enter a ticker symbol.")
            return
        result = lookup_tickers(ticker)  # all the data
        if not result:
            return
        self.search_bar_input.clear()
        self.search_requested.emit(result)


class ModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout

        top_row_layout = QHBoxLayout()
        top_row_layout.addItem(QSpacerItem(40, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

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

        self.setCentralWidget(central_widget)

    def update_status_message(self, message):
        self.result_label = QLabel(message)

    def get_next_day(self):
        print("Fetching model artifacts from S3...")

        MODEL_ARCHIVE_KEY = "pytorch-training-2025-09-22-13-33-36-262/output/model.tar.gz"

        model_buffer = io.BytesIO()

        # Download the entire model.tar.gz archive into the buffer
        s3_client.download_fileobj("sagemaker-us-east-1-307926602475", MODEL_ARCHIVE_KEY, model_buffer)
        model_buffer.seek(0)
        print("Downloaded entire model archive to memory.")

        #torch.serialization.add_safe_globals([numpy.core.multiarray._reconstruct, sklearn.preprocessing.MinMaxScaler, numpy.ndarray])

        # Now open the archive and extract the individual files
        with tarfile.open(fileobj=model_buffer, mode='r:gz') as tar:

            for member in tar.getmembers():
                print(member.name)
            # Load the PyTorch model state dict
            with tar.extractfile('model.pth') as f:
                model_state_dict = torch.load(io.BytesIO(f.read()))
                print("Model state dict loaded successfully.")

            # Load the ticker map
            with tar.extractfile('ticker_to_id.pt') as f:
                ticker_to_id_map = torch.load(io.BytesIO(f.read()))
                print("Ticker map loaded successfully.")

            # Load the scikit-learn scaler object using joblib
            with tar.extractfile("scaler.pt") as f:
                scaler = torch.load(io.BytesIO(f.read()), weights_only=False)
                print("Scaler loaded successfully.")
        #data_handler = DataHandler()
        #list_of_dfs = data_handler.get_dfs_from_s3(prefix="historical_data/")
        #scaler = get_scaler(list_of_dfs)
        # Check for the model's structure. If it's a checkpoint dict, get the state.
        model_state_dict = model_state_dict.get("model_state", model_state_dict)
        if model_state_dict is None:
            raise KeyError("Could not find 'model_state' key in the loaded dictionary.")

        # --- Data fetching and preprocessing for prediction ---
        ticker = self.search_bar_input.text().strip().upper()
        start, end = get_date_range("6M")
        df = fetch_stock_data(ticker, start, end)
        df_engr = feat_engr([df])

        # Scale the input data using the loaded scaler
        feature_columns = ['Close', 'Volume', 'Open', 'High', "Low", "Range", "Delta", "Delta_Percent",
                           "Vol_vs_Avg", "Large_Move", "Large_Up", "Large_Down", "Trend_Up", "Trend_Down",
                           "Break_Up", "Break_Down", "BB_Upper", "BB_Lower", "Cross_BB_Upper",
                           "Cross_BB_Lower", "RSI", "Overbought_RSI", "Oversold_RSI", "Average_Move"]

        print(df_engr)
        df_engr[0][feature_columns] = scaler.transform(df_engr[0][feature_columns])
        #df_engr[feature_columns] = scaler.transform(df_engr[feature_columns])

        tensor = df_to_tensor_with_dynamic_ids(df_engr, ticker_to_id_map)

        if tensor:
            print(f"predicting {ticker.upper()}...")
            # Make sure pred_next_day can handle the scaler
            pred_next_day(tensor[0].unsqueeze(0), ticker_to_id_map, model_state_dict, scaler)

        else:
            print("Could not generate a valid tensor for the given ticker.")

    def train_on_cloud(self):
        try:
            subprocess.Popen([sys.executable, os.path.join('app', 'sage.py')], cwd=os.getcwd())
            print("SageMaker training job started. Check your AWS console for progress.")
        except FileNotFoundError:
            print("Error: sage.py not found. Make sure the file exists.")
        except Exception as e:
            print(f"An error occurred: {e}")
