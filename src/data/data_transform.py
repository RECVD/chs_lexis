import os

import numpy as np
import datetime as dt
import pandas as pd
from pathlib import Path

#############################################
# Functions which apply to all of wrangling #
#############################################

def convert_dates(date):
    """

    :param date: date in the format described above of type str
    :return: The same date in datetime format.
    """
    """
    LexisNexis reports dates in the following format: "201606" means June 2016.  We'll assume that the day is the first,
    and convert to  2016/06/01 in the datetime format.

    Arguments:
        date: date in the format described above of type str
    Returns:
        The same date in datetime format.
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
        date_time = dt.date(year, month, day)
    except ValueError:  # some have months of 00 for some reason, which causes an obvious error
        date_time = dt.date(year, 1, day)

    return date_time


def convert_all_dates(work_history):
    """
    Converts all the columns in work_history containing the word "date" to pandas Timestamps
    """
    copy = work_history.copy()

    date_cols = [col for col in work_history.columns if ('date' in col or 'dt' in col)]

    work_history_dates = work_history[date_cols]
    work_history_dates = work_history_dates.applymap(convert_dates).apply(pd.to_datetime, axis=0)

    copy[date_cols] = work_history_dates

    return copy


def diff_month(date_series_1, date_series_2):
    """
    Computes Elementwise time difference between two series of datetime dates, in months.

    :param date_series_1:
    :param date_series_2:
    :return: Difference between times in months.
    """
    date_series_1_year, date_series_1_month = date_series_1.map(lambda x: x.year), date_series_1.map(lambda x: x.month)
    date_series_2_year, date_series_2_month = date_series_2.map(lambda x: x.year), date_series_2.map(lambda x: x.month)

    return abs((date_series_2_year - date_series_1_year) * 12 + date_series_2_month - date_series_1_month)


##########################################
# Type-Specific Data-Wrangling Functions #
##########################################


def clean_add_history(lexis_address_filename):
    """

    :param lexis_address_filename: the filename string for the LexisNexis address history, relatives and associates file
    :return: A dataframe with the following variables:

    """
    pass

def clean_license_history(work_history_filename):
    """
    Computes single patient-level derived variables for work history.  These include:
        - lex_professional_c: Count of professional licenses ever held
        - lex_professional_any:  Binary indicator variable.  Denotes whether the subject has ever held any professional
            licenses.

    :param work_history_filename: the filename string for the People at Work LexisNexis file
    :return: Pandas.DataFrame().  Index is ssn-altkey, with two columns:  lex_professional_c and lex_professional_any.
    """

    def get_lex_professional_any(work_history):
        """
        Returns a pandas series by the ssn-altkey index that contains 0 if the given subject had no professional
        licenses, or 1 if they had at least one professional license.

        :param work_history: The LexisNexis work history file as a Pandas DataFrame object.  Index should be
        ssn-altkey.
        :return: Pandas series denoting the lex_professional_any variable.  Index is still ssn-altkey.
        """
        # drop several columns that always contain data (even if no license is present)
        license_present = work_history.drop(['gender', 'yrdeath', 'city', 'state'], axis=1) \
            .notnull() \
            .any(axis=1) \
            .groupby(level=0) \
            .any() \
            .rename('lex_professional_any')

        return license_present.astype("int64")


    def get_lex_professional_c(work_history, license_present):
        """
        Returns a pandas series by the ssn-altkey index that contains the number of professional licenses a given
        subject has ever held.

        :param work_history: The LexisNexis work history file as a Pandas DataFrame object.  Index should be
        ssn-altkey.
        :param license_present: Integer indicator variable of license being present, output of
        get_lex_professional_any().
        :return: Pandas series denoting the lex_professional_c variable. Index is still ssn-altkey.
        """
        # Create a series for numlicenses with index 'ssn_altkey', one row per participant.
        # Index will not match up with original data file, which had at least one record per participant, with more
        # existing in the case of multiple licenses
        license_c = license_present.copy(deep=True)

        license_c[license_c == 1] = work_history.drop(['gender', 'yrdeath', 'city', 'state'], axis=1) \
            .notnull() \
            .any(axis=1) \
            .groupby(level=0) \
            .size() \
            .rename('lex_professional_c')

        return license_c

    work_history = pd.read_csv(work_history_filename, index_col='ssn_altkey') \
        .drop_duplicates()
    work_history = convert_all_dates(work_history)

    lex_professional_any = get_lex_professional_any(work_history)
    lex_professional_c = get_lex_professional_c(work_history, lex_professional_any)

    return pd.concat([lex_professional_any, lex_professional_c], axis=1)


def clean_work_history(license_history_filename):
    """

    :param license_history_filename:
    :return:
    """
    pass

def clean_vote_history(vote_history_filename):
    """

    :param vote_history_filename:
    :return:
    """
    pass


def clean_property_history(property_history_filename):
    """

    :param property_history_filename:
    :return:
    """

    # drop non-time series variables
    prop = pd.read_csv(property_history_filename) \
        .drop(columns=['gender', 'yrdeath', 'city', 'state', 'record_type'])

    stubnames = ['Prop-city_', 'Prop-state_', 'Assessed-value_', 'Total-value_', 'Sale-date_', 'Sale-price_',
                 'Mortgage-amount_', 'Total-market-value_']

    # size of each non-empty ssn-altkey group
    prop_c_nonzero = pd.wide_to_long(prop, stubnames=stubnames, i="ssn_altkey", j="time_series") \
        .sort_index() \
        .dropna(how='all') \
        .groupby(level=0) \
        .size()

    # Join in to complete index including all ssn-altkey values
    prop_c = prop_c_nonzero.combine_first(pd.Series(0, index=prop["ssn_altkey"])) \
        .astype("int64") \
        .rename('lex_propertyown_c')

    return prop_c


if __name__ == "__main__":
    # define paths for data reading based on project structure
    cwd = os.getcwd()
    proj_root = Path(cwd).parent.parent
    data_path = proj_root / 'data' / 'raw'

    license_history_filename = data_path / 'LN_Output_ProfLicenses_LN_InputLexisNexisCHSParticipantsNS.Dataset.csv'
    property_history_filename = data_path / 'LN_Output_Property_LN_InputLexisNexisCHSParticipantsNS.Dataset.csv'

    print(clean_property_history(property_history_filename))







