from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from app.ui import MainWindow
import data as data_handler
from search import get_date_range

## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

def main():
    print("Starting app...")
    #initialize_database()
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


S3_BUCKET_NAME = "stock-screener-bucker"


def initialize_database():
    tickers = data_handler.get_sp500_tickers()

    end_date, start_date = get_date_range("3Y")

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            stock_data = data_handler.fetch_stock_data(ticker, start_date, end_date)

            s3_path = f"historical_data/{ticker.lower()}.parquet"

            data_handler.save_to_s3(stock_data, S3_BUCKET_NAME, s3_path)

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")


if __name__ == "__main__":
    main()