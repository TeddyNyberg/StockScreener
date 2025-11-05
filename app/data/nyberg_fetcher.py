import pandas as pd
from app.utils import get_date_range


def get_nyberg_price():
    data = pd.read_excel("backtest_results_jan.xlsx", sheet_name="Summary_Performance")
    return data.loc[data.index[-1], 'Total_Value_At_Close']

def get_nyberg_data(time):

    start_time, end_time = get_date_range(time)

    data = pd.read_excel("backtest_results_jan.xlsx", sheet_name="Summary_Performance")
    data.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    data.set_index("Date", inplace=True)

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
