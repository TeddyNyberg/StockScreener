from config import *
from app.data.ticker_source import get_sp500_tickers
from app.ml_logic.strategy import calculate_kelly_allocations_new
from app.data.yfinance_fetcher import get_historical_data
import pandas as pd
from app.utils import get_date_range
from app.ml_logic.model_loader import load_model_artifacts_local
from app.ml_logic.pred_models.only_close_model import setup_pred_model
from app.ml_logic.strategy import optimal_picks_new
from datetime import datetime
import os
import numpy as np

import time

## pyinstaller app/main.py --onedir --name stock_screener

## alt do --onefile for cleaner exp but slowwwwwwwwwwww

## docker-compose up -d
## --build

def main():


    sp_tickers = get_sp500_tickers(test=True)
    start, end = get_date_range("1.5Y", None)
    processed_tickers = [t.replace(".", "-") for t in sp_tickers]
    all_historical_data = get_historical_data(processed_tickers, start, end)

    dict, config = load_model_artifacts_local(MODEL_MAP["D"]["model_filepath"])
    # calculate_kelly_allocations(MODEL_MAP["A"]["prefix"])

    cur_value = 100000



    """start = time.perf_counter()
    for i in range(100):
        calculate_kelly_allocations_new("A", False, all_historical_data=all_historical_data)
    end = time.perf_counter()
    print("Elapsed pd: ", end - start)

    start = time.perf_counter()
    for i in range(100):
        calculate_kelly_allocations_new("A", False, all_historical_data=all_historical_data)
    end = time.perf_counter()
    print("Elapsed np: ", end - start)"""


    # use 1/28/2025
    date_range = pd.date_range("1/28/2025", pd.to_datetime(datetime.now().strftime('%m/%d/%Y')) - pd.Timedelta(days=1), freq='B')
    portfolio_df = pd.DataFrame(index=date_range, columns=['Total_Value_At_Close', 'Cash_At_Open'], dtype=float)
    portfolio_df.index.name = "Date"
    portfolio_df.loc[date_range[0], 'Total_Value_At_Close'] = 100000
    portfolio_df.loc[date_range[1], "Cash_At_Open"] = 100000
    missed_day = False
    for i in range(1, len(date_range)):
        current_day = date_range[i]
        prev_day = date_range[i - 1]
        count = 0
        while missed_day:
            try:
                x = all_historical_data["Close"]["GOOG"][prev_day]
                missed_day = False
            except:
                prev_day = date_range[i - 1 - count]
                count+=1

        try:
            final_allocs, _ = calculate_kelly_allocations_new("D", False, end=prev_day,
                                                              all_historical_data=all_historical_data)

            portfolio_return = 0
            for ticker, alloc, _ in final_allocs:
                gain = alloc * (
                        all_historical_data["Close"][ticker][current_day] - all_historical_data["Close"][ticker][
                    prev_day]) / all_historical_data["Close"][ticker][prev_day]
                portfolio_return += gain
            cur_value += cur_value * portfolio_return
            portfolio_df.loc[current_day, 'Total_Value_At_Close'] = cur_value
        except:
            print("skip calc kelly on ", prev_day)
            missed_day = True
            continue

    portfolio_df[['Total_Value_At_Close']] = \
        (np.floor(portfolio_df[['Total_Value_At_Close']] * 1000) / 1000)

    file_path = MODEL_MAP["D"]["csv_filepath"]
    #mode = "a" if os.path.exists(file_path) else "w"
    mode = "w"
    header = not os.path.exists(file_path)
    portfolio_df.to_csv(file_path, mode=mode, header=header, date_format="%m/%d/%Y")


    """
    # len(date_range)
    avg_ret_each_pos = [0.0] * 30
    num_trading_days = 0
    returns_series = []
    top30_gains_df = pd.DataFrame()
    for i in range(len(date_range)):
    #for i in range(1,5):
        current_day = date_range[i]
        prev_day = date_range[i - 1]

        # if end=None, end=today 12:00:00 AM
        final_allocs, _ = calculate_kelly_allocations_new("A", False, end=prev_day, all_historical_data=all_historical_data)

        try:
            portfolio_return = 0
            for ticker, alloc, _ in final_allocs:
                gain = alloc * (
                            all_historical_data["Close"][ticker][current_day] - all_historical_data["Close"][ticker][
                        prev_day]) / all_historical_data["Close"][ticker][prev_day]
                portfolio_return += gain
            returns_series.append({'Date': current_day, 'Portfolio_Return': portfolio_return})

        except KeyError as e:
            print(
                f"Skipping day {current_day.strftime('%Y-%m-%d')}: Missing data for date {e}. Likely a weekend or holiday.")
            continue

        print("BOUGHT ", prev_day, " SOLD ", current_day, " RETURNS ", portfolio_return)

        top30_data = final_allocs[:30]
        print("TOP30: ", top30_data)
        top30 = pd.DataFrame(top30_data, columns=['Ticker', 'Allocation', 'Mu'])
        top30['Position_Rank'] = [f"Pos_{r + 1:02d}" for r in top30.index]
        top30['Date'] = current_day

        def calculate_weighted_gain(row):
            ticker = row['Ticker']
            alloc = row['Allocation']
            price_current = all_historical_data["Close"][ticker][current_day]
            price_prev = all_historical_data["Close"][ticker][prev_day]

            return alloc * (price_current - price_prev) / price_prev

        top30['Gain'] = top30.apply(calculate_weighted_gain, axis=1)
        pivot_top30 = top30.pivot(index='Date', columns='Position_Rank', values='Gain')

        top30_gains_df = pd.concat([top30_gains_df, pivot_top30])

        j = 0
        for index, row in top30.iterrows():
            gain_pos = row['Gain']
            avg_ret_each_pos[j] += gain_pos
            j += 1

        num_trading_days += 1

        print("arep: ", avg_ret_each_pos)



    Final_Avg_Ret_Pos = np.array(avg_ret_each_pos) / num_trading_days
    print(Final_Avg_Ret_Pos)
    returns_df = pd.DataFrame(returns_series).set_index('Date')

    combined_df = returns_df.join(top30_gains_df, how='inner')
    correlation_matrix = combined_df.corr()

    correlation_with_return = correlation_matrix['Portfolio_Return'].drop('Portfolio_Return')

    print("Correlation of Top 30 Position Weights with Daily Portfolio Return")
    print(correlation_with_return.to_markdown(numalign="left", stralign="left"))

    """


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

