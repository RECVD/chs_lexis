# Imports
try:
    import numpy as np
except ImportError:
    raise ImportError('Not able to load numpy module')
try:
    import datetime as datetime
except ImportError:
    raise ImportError('Not able to load datetime module')
try:
    import pandas as pd
except ImportError:
    raise ImportError('Not able to load pandas module')


# Functions
def convert_dates(date):
    """
    Function to convert the weird no space LexisNexis Dates to DateTime format
    """
    try:
        date = str(int(date))
    except ValueError:  # Can't convert to int because NaN
        return np.nan

    try:
        year = int(date[:4])
    except ValueError:  # no value exists for year (less than four characters), return NaN
        return np.nan
    try:
        month = int(date[4:6])
    except ValueError:  # no value exists for month, assume January
        month = 1
    try:
        day = int(date[6:])
        if not day:
            day = 1
    except ValueError:  # no value exists for day, assume the first
        day = 1

    try:
        dt = datetime.date(year, month, day)
    except ValueError: #some have months of 00 for some reason, which causes an obvious error
        dt = datetime.date(year, 1, day)
    return dt

def convert_all_dates(df):
    """
    Converts all the columns in df containing the word "date" to pandas Timestamps
    """
    copy = df.copy()

    date_cols = [col for col in df.columns if ('date' in col or 'dt' in col)]

    df_dates = df[date_cols]
    df_dates = df_dates.applymap(convert_dates).apply(pd.to_datetime, axis=0)

    copy[date_cols] = df_dates

    return copy