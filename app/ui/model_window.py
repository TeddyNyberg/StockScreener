from PySide6.QtWidgets import (QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout,
                               QLineEdit, QPushButton, QSpacerItem, QGridLayout)

from app.utils import get_date_range
from app.data.yfinance_fetcher import get_historical_data
from app.ml_logic.strategy import calculate_kelly_allocations, predict_single_ticker
from app.ml_logic.tester import continue_backtest
import subprocess
import sys
from settings import *
from config import *
from app.ui.scatter_canvas import MplCanvas, create_return_figure


class ModelWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Model")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.layout = layout

        top_row_layout = QHBoxLayout()

        #fine_tune_btn = QPushButton("Fine Tune")
        #fine_tune_btn.clicked.connect(lambda: tune())  tune from ml_logic.strategy
        #top_row_layout.addWidget(fine_tune_btn)

        next_day_picks = QPushButton("Next Day Picks")
        next_day_picks.clicked.connect(self.show_kelly_bet)
        top_row_layout.addWidget(next_day_picks)


        #TODO: make this filename universal?
        back_test_button = QPushButton("Back Test")
        back_test_button.clicked.connect(lambda: continue_backtest("A"))
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

        self.fourth_layout = QHBoxLayout()
        self.plot_button = QPushButton("Show Return Scatterplot")
        self.plot_button.clicked.connect(self.show_scatterplot)
        self.fourth_layout.addWidget(self.plot_button)
        layout.addLayout(self.fourth_layout)

        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_container)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_container)

        self.setCentralWidget(central_widget)



    def update_status_message(self, message):
        self.result_label = QLabel(message)

    # This will only be used w base model
    def get_next_day(self):
        ticker = self.search_bar_input.text().strip().upper()
        if not ticker:
            self.update_status_message("Please enter a ticker symbol.")
            return
        try:
            prediction, latest_close_price = predict_single_ticker(ticker)

            print(f"most recent day {latest_close_price}")
            print(f"Predicted next day value: {prediction}")

        except Exception as e:
            self.update_status_message(f"Error predicting for {ticker}: {e}")
            print(f"Prediction error: {e}")

    def show_kelly_bet(self):
        final_allocations, _ = calculate_kelly_allocations("A", False)

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

        start, end = get_date_range("1M")
        goog = get_historical_data("GOOG", start, end) # sanity check, shows latest data available for model
        print(goog.tail(1))

    def show_scatterplot(self):
        ticker = "NYBERG-A"
        for i in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        fig = create_return_figure(ticker, "NYBERG-B")

        if fig is not None:
            canvas = MplCanvas(fig)
            self.plot_layout.addWidget(canvas)
            self.resize(800, 650)
            print(f"Scatterplot updated for {ticker} vs SPY.")
        else:
            print(f"Could not generate plot for {ticker}.")


def train_on_cloud():
    try:
        subprocess.Popen([sys.executable, os.path.join('app', 'sage.py')], cwd=os.getcwd())
        print("SageMaker training job started. Check your AWS console for progress.")
    except FileNotFoundError:
        print("Error: sage.py not found. Make sure the file exists.")
    except Exception as e:
        print(f"An error occurred: {e}")