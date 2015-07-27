import unittest
import os
import sys
import datetime

import pandas as pd
from pandas_datareader import data, wb
#import factories


class DataCollectionTest(unittest.TestCase):
    
    def setUp(self):
        # create test data directory
        self.data_dir = "data_tests"

    def test_data_directory_exists(self):
        self.assertEqual(os.path.exists(self.data_dir), True)

    def test_pandas_can_write_to_data_dir(self):
        df = pd.DataFrame(data=[1,2,3])
        df.to_csv('{0}/test.csv'.format(self.data_dir))
        self.assertEqual(os.path.isfile('{0}/test.csv'.format(self.data_dir), True))
        # add logic to delete file

    def test_pandas_datareader_connection(self):
        start = datetime.datetime(2015, 1, 1)
        end = datetime.datetime(2015, 2, 1)
        df = data.DataReader(["F",], 'yahoo', start, end)

    def test_scrapy_app(self):
        assert False, "TODO: finish me"

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()