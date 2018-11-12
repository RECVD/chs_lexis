import os

import numpy as np
import datetime as dt
import pandas as pd

from scipy.sparse.csgraph import connected_components
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

def wrangle_license_history(license_history_filename):
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


#############################################
######### Work History Cleaning #############
#############################################

def reductionFunction(data):
    """Reduces overlapping date intervals into non-overlapping date intervals

    This is accomplished by a representing the data as an undirected graph for each ssn-altkey.  Each date is a vertex
    in the graph, and the graphs are bipartite with the two disjoint sets being start and end dates.  If a start date is
    before an end date, there is an edge between those two vertices.  We then find all connected components of the
    graph.  Each connected component will be one date range. We take the earliest of the start dates and the latest of
    the end dates as the beginning and ending date for each date range.

    Keyword Arguments:
        data: A pandas DataFrame with first_seen_date and last_seen_date columns. Meant to only operate on a single
        participant at a time, if used on multiple participants should be implemented with pandas.groupby.apply()
    Returns: A pandas dataframe in the long format with a first_seen_date and last_seen_date for each connected
        component, numerically indexed by "num"
    """
    # create a 2D graph of connectivity between date ranges
    start = data.first_seen_date.values
    end = data.last_seen_date.values
    graph = (start <= end[:, None]) & (end >= start[:, None])

    # find connected components in this graph
    n_components, indices = connected_components(graph)
    indices += 1

    # group the results by these connected components
    data_reduced = data.groupby(indices).agg({'first_seen_date': 'min',
                                              'last_seen_date': 'max',
                                              'num': 'first'})

    return data_reduced


def clean_work_history(work_history_uncleaned):
    """ Formatting for the work_history filename

    Change column naming convention and subset to only needed columns.

    Keyword Arguments:
        work_history_uncleaned: the raw version of the LexisNexis "People at Work" file
    Returns: cleaned and subsetted version
    """
    # Change column naming convention to match the other files, with chronological number at the end
    cols = work_history_uncleaned.columns.tolist()
    for i, colname in enumerate(cols):
        if 'pawk' in colname:
            shift = colname[:4] + colname[6:] + colname[4:6]
            cols[i] = shift

    work_history_cleaned = work_history_uncleaned.copy()
    work_history_cleaned.columns = cols

    # subset to only the columns needed
    seen_cols = ['ssn_altkey', 'yrdeath'] + [col for col in work_history_cleaned.columns if '_seen' in col]
    work_history_cleaned = work_history_cleaned[seen_cols].reset_index(drop=True) \
        .drop_duplicates(subset='ssn_altkey') #duplicates present for some reason

    return work_history_cleaned


def create_emp_intervals(work_history_cleaned):
    """ Translate the cleaned work history to employment intervals in the long format

    Keyword Arguments:
        work_history_cleaned: The output of clean_work_history

    """
    df_long = pd.wide_to_long(work_history_cleaned, ['pawk_last_seen_', 'pawk_first_seen_'], i='ssn_altkey', j='num')
    df_long = df_long.sort_index().dropna(subset=['pawk_last_seen_', 'pawk_first_seen_'])
    df_long.columns = ['death', 'last_seen_date', 'first_seen_date']

    # Cleaning - convert dates, drop duplicates within groups, and drop records where last_seen_date == first_seen_dat
    df_long = convert_all_dates(df_long) \
        .groupby(level=0) \
        .apply(lambda x: x.drop_duplicates()) \
        .reset_index(level=0, drop=True)

    df_long = df_long[df_long['last_seen_date'] - df_long['first_seen_date'] != dt.timedelta(days=0)]

    df_long = df_long \
        .reset_index(level=1, drop=True) \
        .set_index("first_seen_date", append=True) \
        .sort_index()

    df_long['num'] = df_long.groupby(level=0).cumcount() + 1
    df_long = df_long \
        .reset_index(level=1, drop=False) \
        .set_index('num', append=True)

    return df_long


def get_num_jobs(emp_intervals_long):
    """ Get the number of jobs for each ssn_altkey with >= 1 job, before reduction"""
    return emp_intervals_long.groupby(level=0) \
        .size() \
        .rename("lex_numberofjobs_c") \
        .to_frame()

def reduce_emp_intervals(emp_intervals_long):
    """ Apply the reduction function to each person and return to the wide format"""
    df_long_reduced = emp_intervals_long \
        .reset_index(drop=False) \
        .groupby('ssn_altkey') \
        .apply(lambda x: reductionFunction(x)) \
        .drop(columns=["num"])

    unstacked = df_long_reduced.unstack()
    unstacked.columns = unstacked.columns.map('{0[1]}_emp_range_{0[1]}_{0[0]}'.format)
    unstacked = unstacked.reindex(columns=sorted(unstacked.columns))
    unstacked.columns = [x[2:-5] for x in unstacked.columns]

    return unstacked



def wrangle_work_history(work_history_filename):
    """ Creates all the derived variables from the "People at work" LexisNexis file "

    These includes:
        - lex_numberofjobs_c: total number of jobs held
        - lex_emp_range_*_first_seen: First seen date of the range of employement * where * is an int
        - lex_emp_range_*_last_seen: Last seen date of the  range of emploment * where * is an int

    Returns these variables, with one value for each ssn-altkey
    """
    work_history_uncleaned = pd.read_csv(work_history_filename)
    work_history_cleaned = clean_work_history(work_history_uncleaned)

    emp_intervals_long = create_emp_intervals(work_history_cleaned)

    #create baseline with all ssn-altkeys
    final_df = pd.DataFrame(0, index=work_history_uncleaned.ssn_altkey.unique(), columns=["lex_numberofjobs_c"])
    final_df = get_num_jobs(emp_intervals_long).combine_first(final_df).astype("int64")
    wide_intervals = reduce_emp_intervals(emp_intervals_long)

    return pd.concat([wide_intervals, final_df], axis=1)


def wrangle_vote_history(vote_history_filename):
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

    Keyword Arguments:
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


def wrangle_property_history(property_history_filename):
    """ Creates the derived variable lex_propertyown_c from the "Property" LexisNexis file.

    The function returns a pandas Series lex_propertyown_c, denoting the total number of properties owned by a given
     participant, with ssn-altkey as the index.

    Keywork Arguments:
    property_history_filename -- the filename string for the "Property" LexisNexis file
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
    work_history_filename = data_path / 'LN_Output_Employment_LN_InputLexisNexisCHSParticipantsNS.Dataset.csv'






