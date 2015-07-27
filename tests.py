import unittest
import os
import sys
import datetime

import pandas as pd
from pandas_datareader import data, wb
#import factories

import settings
from utils import load_data as ld


class DataCollectionTest(unittest.TestCase):
    
    def test_datadir_is_set(self):
        try:
            settings.DATA_DIR
        except:
            raise Exception("DATA_DIR not set in settings.py")

    def test_data_directory_exists(self):
        self.assertEqual(os.path.exists(settings.DATA_DIR), True)

    def test_pandas_can_write_to_data_dir(self):
        df = pd.DataFrame(data=[1,2,3])
        df.to_csv('{0}/test.csv'.format(settings.DATA_DIR))
        self.assertEqual(os.path.isfile('{0}/test.csv'.format(settings.DATA_DIR), True))
        # add logic to delete file

    def test_pandas_datareader_connection(self):
        start = datetime.datetime(2015, 1, 1)
        end = datetime.datetime(2015, 2, 1)
        df = data.DataReader(["F",], 'yahoo', start, end)

    def test_scrapy_app(self):
        assert False, "TODO: finish me"

    def tearDown(self):
        pass


class DataLoadTest(unittest.TestCase):

        def setUp(self):
            # create test data directory
            self.data_dir = "{0}/tests".format(settings.DATA_DIR)
            # download small csvs and scrapy data

        def test_file_picker_selects_latest_file(self):
            # make two files, with a, b
            # ensure if b is made after a, that b is chosen
            assert False, "TODO: finish me"

        def test_load_market_data_returns_df(self):
            assert False, "TODO: finish me"

        def test_load_fred_data_returns_df(self):
            assert False, "TODO: finish me"

        # def test_load_fama_data_returns_df(self):
        #     assert False, "TODO: finish me"

        def test_load_multipl_data_returns_df(self):
            assert False, "TODO: finish me"


if __name__ == '__main__':
    unittest.main()