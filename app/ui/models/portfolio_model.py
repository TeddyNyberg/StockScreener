from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtGui import QColor
from decimal import Decimal
from app.data.yfinance_fetcher import get_price


class PortfolioModel(QAbstractTableModel):
    def __init__(self, raw_portfolio_data, cash_balance):
        super().__init__()
        self._headers = ["Ticker", "Price", "Shares", "Cost Basis", "Avg Cost", "P/L", "% Return"]
        self._data = []
        self._total_value = Decimal(0.0)

        self._preprocess_data(raw_portfolio_data, cash_balance)

    def _preprocess_data(self, raw_data, cash_balance):
        total_equity = Decimal(0.0)

        for row in raw_data:
            ticker = row[0]
            shares = int(row[1])
            total_basis = Decimal(str(row[2]))

            try:
                cur_price = Decimal(str(get_price(ticker)))
            except Exception:
                cur_price = Decimal("0.00")

            if shares > 0:
                avg_cost = total_basis / shares
                market_value = cur_price * shares
                profit_loss = market_value - total_basis
                percent_return = (profit_loss / total_basis) * 100 if total_basis != 0 else 0

                total_equity += market_value
            else:
                avg_cost = 0
                profit_loss = 0
                percent_return = 0

            self._data.append({
                "ticker": ticker,
                "price": cur_price,
                "shares": shares,
                "basis": total_basis,
                "avg": avg_cost,
                "pl": profit_loss,
                "pct": percent_return,
                "is_cash": False
            })

        cash_balance = Decimal(str(cash_balance))
        total_equity += cash_balance

        self._data.append({
            "ticker": "CASH",
            "price": Decimal("1.00"),
            "shares": cash_balance,
            "basis": cash_balance,
            "avg": Decimal("1.00"),
            "pl": Decimal("0.00"),
            "pct": Decimal("0.00"),
            "is_cash": True
        })

        self.total_value = total_equity

    def rowCount(self, parent=None):
        return len(self._data)

    def columnCount(self, parent=None):
        return len(self._headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row_item = self._data[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return row_item["ticker"]
            if col == 1: return f"${row_item['price']:,.2f}"
            if col == 2: return str(row_item["shares"])
            if col == 3: return f"${row_item['basis']:,.2f}"
            if col == 4: return f"${row_item['avg']:,.2f}"
            if col == 5: return f"${row_item['pl']:,.2f}"
            if col == 6: return f"{row_item['pct']:.2f}%"

        if role == Qt.ItemDataRole.ForegroundRole:
            if col in [5, 6]:
                if row_item["pl"] > 0:
                    return QColor("green")
                elif row_item["pl"] < 0:
                    return QColor("red")

        return None