import pandas as pd


def is_tuning_day(date, tuning_period):
    date = pd.to_datetime(date)
    if tuning_period == "daily":
        return True
    if tuning_period == 'weekly':
        return date.dayofweek == 0
    return False




