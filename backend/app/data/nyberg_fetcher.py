import pandas as pd
from backend.app.config import MODEL_MAP
from backend.app.utils import get_date_range


def get_nyberg_price(ticker):
    print(ticker, "get_nyberg_price")
    data = _read_nyberg_file(ticker)
    return data.loc[data.index[-1], 'Total_Value_At_Close']

def get_nyberg_data(time, ticker):
    print(ticker, "get_nyberg_data")
    start_time, end_time = get_date_range(time)
    data = _read_nyberg_file(ticker)

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    selected_data = data.loc[start_time:end_time]
    close_values = selected_data["Total_Value_At_Close"]

    nyberg_df = pd.DataFrame({
        'Open': close_values,
        'High': close_values,
        'Low': close_values,
        'Close': close_values,
        'Volume': 0
    })
    nyberg_df.index.name = 'Date'

    return nyberg_df

def get_nyberg_name(ticker):
    version = ticker[7]
    return MODEL_MAP[version]["name"]

def _read_nyberg_file(ticker):
    version = ticker[7]
    filepath = MODEL_MAP[version]["csv_filepath"]
    return pd.read_csv(filepath, index_col=0, parse_dates=True)
