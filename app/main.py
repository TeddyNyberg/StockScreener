from PySide6.QtWidgets import QApplication
from app.ui import MainWindow


## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

## docker-compose up -d
## --build

def main():
    print("Starting app...")
    #initialize_database() one time use
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()




"""

so s3 has 3years of data, the model is trained on 3 years


def initialize_database():

    data_handler = DataHandler()

    tickers = get_sp500_tickers()

    start_date, end_date = get_date_range("3Y")

    all_stock_data = []

    for ticker in tickers:
        try:
            print(f"Fetching data for {ticker}...")
            stock_data = fetch_stock_data(ticker, start_date, end_date)

            if stock_data is not None:
                all_stock_data.append(stock_data)

                s3_path = f"historical_data/{ticker.lower()}.parquet"
                data_handler.save_to_s3(stock_data, S3_BUCKET_NAME, s3_path)
            else:
                print(f"Skipping {ticker} due to no data.")

        except Exception as e:
            print(f"Failed to process {ticker}: {e}")

    print("Finished fetching and saving data. Returning the list of DataFrames.")
    return all_stock_data
"""

if __name__ == "__main__":
    main()

