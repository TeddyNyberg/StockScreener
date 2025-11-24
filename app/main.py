from config import *
from app.data.ticker_source import get_sp500_tickers

## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

## docker-compose up -d
## --build

def main():
    from app.ml_logic.strategy import calculate_kelly_allocations_new
    from app.data.yfinance_fetcher import get_historical_data
    from app.data.ticker_source import get_sp500_tickers
    from app.utils import get_date_range
    from app.ml_logic.model_loader import load_model_artifacts
    from app.ml_logic.strategy import optimal_picks_new

    import time

    sp_tickers = get_sp500_tickers(test=True)
    start, end = get_date_range(lookback_period, None)
    processed_tickers = [t.replace(".", "-") for t in sp_tickers]
    all_historical_data = get_historical_data(processed_tickers, start, end)
    load_model_artifacts()
    # calculate_kelly_allocations(MODEL_MAP["A"]["prefix"])

    start = time.perf_counter()
    for i in range(100):
        calculate_kelly_allocations_new("A", False, all_historical_data=all_historical_data)
    end = time.perf_counter()
    print("Elapsed pd: ", end - start)

    start = time.perf_counter()
    for i in range(100):
        calculate_kelly_allocations_new("A", False, all_historical_data=all_historical_data)
    end = time.perf_counter()
    print("Elapsed np: ", end - start)

    from PySide6.QtWidgets import QApplication
    from app.ui import start_application
    app = QApplication([])
    start_application()
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

