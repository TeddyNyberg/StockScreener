from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QSpacerItem, QSizePolicy
from .search import lookup_ticker, get_chart
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


class DetailsWindow(QMainWindow):
    def __init__(self, ticker_data):
        super().__init__()
        self.setWindowTitle(f"Details for {ticker_data["ticker"]}")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        layout.addWidget(QLabel(f"Name: {ticker_data["name"]}"))
        layout.addWidget(QLabel(f"Price: {ticker_data["price"]} {ticker_data["currency"]}"))

        self.canvas = CustomChartCanvas(ticker_data["chart"])
        layout.addWidget(self.canvas)

        self.setCentralWidget(central_widget)


class CustomChartCanvas(FigureCanvas):
    def __init__(self, chart_figure, parent=None):
        self.fig = chart_figure
        self.axes = self.fig.get_axes()[0]
        super().__init__(self.fig)

        self.setParent(parent)
        self.mpl_connect('motion_notify_event', self.on_hover)

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
            data = self.fig.data[0]

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