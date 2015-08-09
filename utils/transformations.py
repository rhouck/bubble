import numpy as np
import pandas as pd


def get_max_loss(per, ret):
    """Given a dataframe of increasingly lagged returns in each column, 
    find the largest negative return between the first date in index and `per` periods in the future.

    :param per: An integer specifying number periods into future to consider
    :param ret: A data frame of returns and lagged returns for each date
    :rtype: A float
    """
    m = np.ones([per,per],dtype='bool')
    m[np.triu_indices(per)] = False
    m = np.fliplr(m)
    return ret.ix[:per,1:per].mask(m).min().min()   


def change_space(s, years):
    """Calculates relative increase in value over previous period (increase based on ewma values)"""
    s = pd.ewma(s, halflife=90)
    return s / s.shift(360*years) - 1.


def std_space(s, years):
    return pd.ewmstd(s, halflife=360*years)

def classify_losses(losses, loss_threshold=-.15, period=3):
    """Select loss time frame (`periods`) and classify all negative returns exceeding 
    threshold as 1 and all returns greater than theshold as 0."""
    return losses[period].map(lambda x: 1 if x < loss_threshold else 0)
