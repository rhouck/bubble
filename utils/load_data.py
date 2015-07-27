import os

import pandas as pd

import settings

# market = pd.read_csv('data/market_close.csv', index_col=0, parse_dates=0)
# market.plot(figsize=[15,3])

def get_latest_file_containing_string(path, string):
    """Returns name of file most recently modified that contains the string `string`.
    If no file name contains `string`, None is returned.

    :param path: A string, the path to search
    :param string: A string, the string that must be contained in file name 
    """
    return "hi"

def load_market_data():
    #get_latest_file_containing_string(path, string)
    df = pd.read_csv('{0}/market_close.csv'.format(settings.DATA_DIR), 
                     index_col=0, 
                     parse_dates=0)
    return df

    
if __name__ == '__main__':
    print get_latest_file_containing_string('../data', '7')
