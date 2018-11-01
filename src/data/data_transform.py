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

def clean_license_history(license_history_filename):
    """
    Computes single patient-level derived variables for license history.  These include:
        - lex_professional_c: Count of professional licenses ever held
        - lex_professional_any:  Binary indicator variable.  Denotes whether the subject has ever held any professional
            licenses.

    :param license_history_filename: the filename string for the People at Work LexisNexis file
    :return: Pandas.DataFrame().  Index is ssn-altkey, with two columns:  lex_professional_c and lex_professional_any.
    """

    def get_lex_professional_any(license_history):
        """
        Returns a pandas series by the ssn-altkey index that contains 0 if the given subject had no professional
        licenses, or 1 if they had at least one professional license.

        :param license_history: The LexisNexis license history file as a Pandas DataFrame object.  Index should be
        ssn-altkey.
        :return: Pandas series denoting the lex_professional_any variable.  Index is still ssn-altkey.
        """
        # drop several columns that always contain data (even if no license is present)
        license_present = license_history.drop(['gender', 'yrdeath', 'city', 'state'], axis=1) \
            .notnull() \
            .any(axis=1) \
            .groupby(level=0) \
            .any() \
            .rename('lex_professional_any')

        return license_present.astype("int64")


    def get_lex_professional_c(license_history, license_present):
        """
        Returns a pandas series by the ssn-altkey index that contains the number of professional licenses a given
        subject has ever held.

        :param license_history: The LexisNexis license history file as a Pandas DataFrame object.  Index should be
        ssn-altkey.
        :param license_present: Integer indicator variable of license being present, output of
        get_lex_professional_any().
        :return: Pandas series denoting the lex_professional_c variable. Index is still ssn-altkey.
        """
        # Create a series for numlicenses with index 'ssn_altkey', one row per participant.
        # Index will not match up with original data file, which had at least one record per participant, with more
        # existing in the case of multiple licenses
        license_c = license_present.copy(deep=True)

        license_c[license_c == 1] = license_history.drop(['gender', 'yrdeath', 'city', 'state'], axis=1) \
            .notnull() \
            .any(axis=1) \
            .groupby(level=0) \
            .size()

        return license_c.rename('lex_professional_c')

    license_history = pd.read_csv(license_history_filename, index_col='ssn_altkey') \
        .drop_duplicates()
    license_history = convert_all_dates(license_history)

    lex_professional_any = get_lex_professional_any(license_history)
    lex_professional_c = get_lex_professional_c(license_history, lex_professional_any)

    return pd.concat([lex_professional_any, lex_professional_c], axis=1)


def clean_work_history(work_history_filename):
    """

    """
    pass

def clean_vote_history(vote_history_filename):
    """ Creates all the derived variables from the "Voter" LexixsNexis file.

    These include:
        - lex_votetotal_c: total number of times each person voted
        - lex_voteprim_c: total number of times each person voted in a primary election
        - lex_votegen_c: total number of times each person voted in a general election
        - lex_votepres_c: total number of times each person voted in a presidential election
        - yrdeath: year the person died, included for context in death_vote variable
        - lex_vote_most_recent: most recent time the person voted, included for context in death_vote variable
        - lex_deathvote: indicator variable for whether is listed as voting after their CHS date of death

    The function returns a pandas DataFrame with all these variables, and a unique ssn-altkey as the index.

    Keywork Arguments:
    vote_history_filename -- the filename string for the "Voter" LexisNexis file
    """
    def tot_votes(vote_cleaned):
        """ Returns the total number of times each subject voted as a series with index ssn-altkey.
        Acts on the cleaned dataset.
        """
        return vote_cleaned.apply(sum, axis=1).rename('lex_votetotal_c')

    def prim_votes(vote_cleaned):
        """
        Returns the total number of times each subject voted in a primary election as a series with index ssn-altkey
        Acts on the cleaned dataset.
        """
        prim = vote_cleaned[[x for x in vote_cleaned.columns.tolist() if 'primary' in x]]
        return prim.apply(sum, axis=1).rename('lex_voteprim_c')

    def gen_votes(vote_cleaned):
        """
        Returns the total number of times each subject voted in a general election as a series with index ssn-altkey.
        Acts on the cleaned dataset.
        """
        gen = vote_cleaned[[x for x in vote_cleaned.columns.tolist() if 'general' in x]]
        return gen.apply(sum, axis=1).rename('lex_votegen_c')

    def pres_votes(vote_cleaned):
        """
        Returns the total number of times each subject voted in a presidential election as a series with
        index ssn-altkey.
        Acts on the cleaned dataset.
        """
        pres = vote_cleaned[[x for x in vote_cleaned.columns.tolist() if 'pres' in x]]
        return pres.apply(sum, axis=1).rename('lex_votepres_c')

    def death_votes(vote_uncleaned):
        """ Returns the following variables, acting on the uncleaned dataset:
            - yrdeath
            - lex_most_recent_vote
            - deathvote
        """
        # 'voted_year_1' is the most recent vote
        datecols = ['yrdeath', 'voted_year_1']
        death = vote_uncleaned.groupby('ssn_altkey').last()[datecols]
        death.columns = ['yrdeath', 'lex_most_recent_vote']
        death['lex_deathvote'] = death['yrdeath'] > death['lex_most_recent_vote']

        # so the most_recent_vote is printed without the decimal points
        pd.options.display.float_format = '{:.0f}'.format

        return death

    def clean_vote_data(vote_uncleaned):
        """ Prepares the vote data for summing totals votes on the elections of interest.
        - Subsets to only elections of interest
        - Collapse to only a single row for each ssn-altkey
        - Fill all letters with a positive indicator for voting

        Returns this data in the form of a Pandas DataFrame.

        Keyword Arguments:
        vote_uncleaned -- the original version of the LexisNexis Voter data with numeric index.
        """
        # subset to of interest columns
        vote_terms = ['ssn_altkey', 'special', 'primary', 'general', 'pres', 'other']
        vote_subset = vote_uncleaned[[x for x in vote_uncleaned.columns if any(y in x for y in vote_terms)]] \
            .dropna(axis=1, how="all")

        # Replace NAN with 0 (no vote), all other data with 1 (vote)
        vote_num = vote_subset.groupby('ssn_altkey') \
            .ffill() \
            .groupby('ssn_altkey') \
            .last() \
            .replace(np.nan, '0') \
            .replace(['Y', 'R', 'D', 'A', 'E', 'P', 'M', 'Q', 'U'], "1")  \
            .astype('int64')

        return vote_num

    vote_uncleaned = pd.read_csv(vote_history_filename)
    vote_clean = clean_vote_data(vote_uncleaned)

    df_final = pd.concat([tot_votes(vote_clean),
                          prim_votes(vote_clean),
                          gen_votes(vote_clean),
                          pres_votes(vote_clean),
                          death_votes(vote_uncleaned)], axis=1)

    return df_final


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

    # create derived variables
    license_history_filename = data_path / 'LN_Output_ProfLicenses_LN_InputLexisNexisCHSParticipantsNS.Dataset.csv'
    property_history_filename = data_path / 'LN_Output_Property_LN_InputLexisNexisCHSParticipantsNS.Dataset.csv'
    vote_history_filename = data_path / "LN_Output_Voter_LN_InputLexisNexisCHSParticipantsNS.Dataset.csv"







