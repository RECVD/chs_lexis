import os

import pandas as pd
from pathlib import Path


class Tester(object):
    """ Object to do tests on our data

    """

    def __init__(self):
        # initialize paths for data reading
        cwd = os.getcwd()
        proj_root = Path(cwd).parent
        self.data_path = proj_root / 'data' / 'interim'
        self.write_path = proj_root / 'data' / 'final'

        self.add_history_buffer = self.read_data("add_history_buffer.csv", ind = ['ssn_altkey', 'best_address_num'])
        self.add_history_participant = self.read_data("add_history_participant.csv")
        self.license_history = self.read_data("license_history.csv")
        self.property_history = self.read_data("property_history.csv")
        self.vote_history = self.read_data("vote_history.csv")
        self.work_history = self.read_data("work_history.csv")

    def read_data(self, filename, ind="ssn_altkey"):
        """ Create data frames based on object inpupt files

        """
        return pd.read_csv(self.data_path / filename, index_col=ind)


    def test_license_history(self):
        return len(self.license_history)

    def test_property_history(self):
        return len(self.property_history)

    def test_vote_history(self):
        return len(self.vote_history)

    def test_work_history(self):
        return len(self.work_history)

    def test_address_participant_history(self):
        return len(self.add_history_participant)

    def test_address_buffer_history(self):
        return len(self.add_history_buffer)


if __name__ == "__main__":
    testy = Tester()

    all_participant = testy.license_history.join(testy.property_history, how="outer") \
        .join(testy.vote_history, how="outer") \
        .join(testy.work_history, how="outer") \
        .join(testy.add_history_participant, how="outer") \
        .rename_axis("prs_chs_ssnAltkey_u_2014") \
        .to_csv('lexis_derived_participant.csv')

    testy.add_history_buffer \
        .rename_axis(["prs_chs_ssnAltkey_u_2014", "adr_lex_bestAddressNum_i_2014"]) \
        .to_csv('lexis_derived_buffer.csv')
