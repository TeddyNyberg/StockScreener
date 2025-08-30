from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout,
                               QLineEdit, QPushButton, QSpacerItem, QTableWidget, QTableWidgetItem,
                               QSizePolicy)
from .search import lookup_tickers, get_chart, get_financial_metrics, get_balancesheet, get_info
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd


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

        #self.make_buttons()
        self.make_buttons_v2(["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"], self.update_chart)
        # TODO: make buttons for fin vs info vs inc
        self.show_fin_info("financials")
        self.show_fin_info("balance_sheet")
        self.show_fin_info("info")

        self.setCentralWidget(central_widget)


    def update_chart(self, time, tickers=None):
        # TODO: implement cache
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

    def make_buttons(self):
        button_layout = QHBoxLayout()
        timeframes = ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "5Y", "MAX"]
        for tf in timeframes:
            btn = QPushButton(tf)
            btn.clicked.connect(lambda _, t=tf: self.update_chart(t, [self.ticker_data[0]["ticker"], self.comp_ticker]))
            button_layout.addWidget(btn)
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding)
        button_layout.addItem(spacer)
        self.layout.addLayout(button_layout)

    def make_buttons_v2(self, labels, func):
        button_layout = QHBoxLayout()
        for label in labels:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, t=label: func(t, [self.ticker_data[0]["ticker"], self.comp_ticker]))
            button_layout.addWidget(btn)
        spacer = QSpacerItem(40,20, QSizePolicy.Expanding)
        button_layout.addItem(spacer)
        self.layout.addLayout(button_layout)


    def show_fin_info(self, metrics):
        if metrics == "financials":
            self.layout.addWidget(QLabel("--- Financials ---"))
            df = get_financial_metrics(self.ticker_data[0]["ticker"])
        elif metrics == "balance_sheet":
            self.layout.addWidget(QLabel("--- Balance Sheet ---"))
            df = get_balancesheet(self.ticker_data[0]["ticker"])
        elif metrics == "info":
            self.layout.addWidget(QLabel("--- Info ---"))
            info = get_info(self.ticker_data[0]["ticker"])
            df = pd.DataFrame(info.items(), columns=["Metric", "Value"])
            table_widget = QTableWidget()
            table_widget.setRowCount(df.shape[0])
            table_widget.setColumnCount(df.shape[1])
            for row_index in range(df.shape[0]):
                for col_index in range(df.shape[1]):
                    value = df.iloc[row_index, col_index]
                    table_widget.setItem(row_index, col_index, QTableWidgetItem(str(value)))
            self.layout.addWidget(table_widget)
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
            self.layout.addWidget(table_widget)



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


    # TODO: show_balncesheet
    # def show_balancesheet(self):
    #    pass

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



