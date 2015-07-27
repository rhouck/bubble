import os
import sys

import pandas as pd

sys.path.append('../')
import settings


def get_latest_file_containing_string(path, string):
    """Returns name of file most recently modified that contains the string `string`.
    If no file name contains `string`, None is returned.

    :param path: A string, the path to search
    :param string: A string, the string that must be contained in file name 
    """
    return "hi"

def latest_csv_to_pandas(string):
    """Returns DF of pandas exported data from latest modified csv file matching string"""
    #get_latest_file_containing_string(path, string)
    # FIX THIS
    fn = "{0}_2015_07_26.csv".format(string)
    df = pd.read_csv("{0}/{1}".format(settings.DATA_DIR, fn), 
                     index_col=0, 
                     parse_dates=0)
    return df
