import pandas as pd
from pandas.tseries.offsets import DateOffset, Day


def get_date_range(time, today = None):  # must be all caps
    if today is None:
        today = pd.Timestamp.today().normalize()

    offsets = {
        "1D": Day(1),
        "5D": Day(5),
        "1M": DateOffset(months=1),
        "3M": DateOffset(months=3),
        "6M": DateOffset(months=6),
        "1Y": DateOffset(years=1),
        "3Y": DateOffset(years=3),
        "5Y": DateOffset(years=5),
        "YTD": DateOffset(year=today.year, month=1, day=1)
    }
    if time == "MAX":
        start = pd.Timestamp(year=1950, month=1, day=1)
    else:
        start = today - offsets[time]
    return start, today