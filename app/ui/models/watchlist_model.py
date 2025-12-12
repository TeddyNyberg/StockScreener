from PySide6.QtCore import QAbstractTableModel, Qt
from decimal import Decimal
from app.data.yfinance_fetcher import get_price

class WatchlistModel(QAbstractTableModel):
    def __init__(self, watchlist_data):
        super().__init__()
        self._headers = ["Ticker", "Price"]
        self._data = []
        self._preprocess_data(watchlist_data)

    def _preprocess_data(self, raw_data):
        for entry in raw_data:
            ticker = entry[0]
            try:
                price = Decimal(str(get_price(ticker)))
            except Exception:
                price = Decimal("0.00")

            self._data.append({
                "ticker": ticker,
                "price": price
            })

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._headers[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row_item = self._data[index.row()]
            col = index.column()
            if col == 0: return row_item["ticker"]
            if col == 1: return f"${row_item['price']:,.2f}"
        return None

    def get_ticker(self, row_index):
        if 0 <= row_index < len(self._data):
            return self._data[row_index]["ticker"]
        return None