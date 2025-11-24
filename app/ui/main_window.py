
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QSpacerItem, QSizePolicy
from app.search.ticker_lookup import lookup_tickers
from app.ui.chart_canvas import CustomChartCanvas
from app.ui.helpers import open_watchlist, open_portfolio, open_window_from_ticker
from app.ui.model_window import ModelWindow
from app.ui.widgets import SearchWidget


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
