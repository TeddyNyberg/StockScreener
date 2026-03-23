import time
import unittest.mock as mock
import pandas as pd
import os
import sys
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def get_mock_df(*args, **kwargs):
    time.sleep(10)

    start_str = kwargs.get('start') or (args[1] if len(args) > 1 else "2023-01-01")
    end_str = kwargs.get('end') or (args[2] if len(args) > 2 else "2023-01-10")

    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)

    total_days = (end_dt - start_dt).days
    if total_days <= 0:
        total_days = 1

    num_rows = max(1, total_days // 10)

    dates = pd.date_range(start=start_dt, periods=num_rows, freq='D')

    df = pd.DataFrame({
        "Open": np.random.uniform(100, 110, size=num_rows),
        "High": np.random.uniform(110, 120, size=num_rows),
        "Low": np.random.uniform(90, 100, size=num_rows),
        "Close": np.random.uniform(100, 110, size=num_rows),
        "Volume": np.random.randint(1000, 5000, size=num_rows)
    }, index=dates)

    return df

async def mock_cur_price(self, ticker_list):
    print(f"--- MOCK LIVE PRICE FOR: {ticker_list} ---")
    return [150.0] * len(ticker_list)



patcher_yf = mock.patch("backend.app.data.yfinance_fetcher.yf.download", side_effect=get_mock_df)
patcher_yf.start()

patcher_live = mock.patch("backend.app.data.yfinance_fetcher.LiveMarketTable.cur_price", new=mock_cur_price)
patcher_live.start()


from backend.app.main import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

