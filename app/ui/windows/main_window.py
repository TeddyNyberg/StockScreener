
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QWidget, QLabel, QVBoxLayout, QPushButton, QSpacerItem, QSizePolicy,
                               QMenu)
from app.search.ticker_lookup import lookup_tickers
from app.ui.chart_canvas import CustomChartCanvas
from app.ui.window_manager import open_detail_window, open_watchlist, open_portfolio
from app.ui.windows.login_window import LoginWindow
from app.ui.windows.model_window import ModelWindow
from app.ui.widgets import SearchWidget


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stock Screener")
        main_layout = QVBoxLayout()

        self.resize(800, 600)

        self.logged_in = False
        self.username = None

        user_layout = QHBoxLayout()
        self.user_btn = QPushButton("Login")
        user_layout.addStretch(1)
        user_layout.addWidget(self.user_btn)




        self.update_user_ui()

        main_layout.addLayout(user_layout)

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
        self.search_widget.search_requested.connect(open_detail_window)
        self.search_widget.message_displayed.connect(self.update_status_message)
        top_layout.addWidget(self.search_widget, alignment=Qt.AlignmentFlag.AlignTop)

        main_layout.addLayout(top_layout)

        spy, spy_chart, data = lookup_tickers("^SPX")
        self.canvas = CustomChartCanvas(data, spy_chart)
        main_layout.addWidget(self.canvas)

        self.setLayout(main_layout)

    def update_status_message(self, message):
        self.result_label = QLabel(message)

    def handle_model(self):
        if self.model_window is None:
            self.model_window = ModelWindow()
        self.model_window.show()

    def update_user_ui(self):

        if self.logged_in:
            self.user_btn.setText(f"{self.username}")

            menu = QMenu(self)
            logout_action = menu.addAction("Logout")
            logout_action.triggered.connect(self.logout)

            self.user_btn.setMenu(menu)
        else:
            self.user_btn.setText("Login")
            self.user_btn.setMenu(None)
            self.user_btn.clicked.connect(self.open_login_window)

    def open_login_window(self):
        login_dialog = LoginWindow()
        if login_dialog.exec():
            print("execed")
            username = login_dialog.get_username()

            if username:
                self.username = username
                self.logged_in = True
                self.update_user_ui()
                print(f"Logged in as {self.username}")


    def logout(self):
        self.logged_in = False
        self.username = None
        self.update_user_ui()
        print("log out")

