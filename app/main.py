from app.ml_logic.tester import continue_backtest

def background_backtesting():
    continue_backtest("A")
    continue_backtest("B", "weekly")
    continue_backtest("C")

def main():
    print("IN MAIN")
    #This will chnage eventually
    background_backtesting()
    # backtest_thread = threading.Thread(target=background_backtesting, args=[])
    # backtest_thread.daemon = True
    # backtest_thread.start()

    from PySide6.QtWidgets import QApplication
    from app.ui import start_application

    app = QApplication([])
    start_application()
    app.exec()


if __name__ == "__main__":
    main()






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

## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

## docker-compose up -d
## --build