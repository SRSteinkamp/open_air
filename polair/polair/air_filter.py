import numpy as np
import pandas as pd
from scipy.signal import convolve
import warnings


def create_lags(feed: pd.DataFrame, features: list, lags):
    """
    Important: this will only work for single feeds so far.
    Index is assumed to be in logical time steps
    """
    feed_out = feed.copy()

    for feat in features:
        for lg in range(1, lags + 1):
            feed_out[f"{feat}_lg{lg}"] = 0
            feed_out.loc[lg:, f"{feat}_lg{lg}"] = feed_out.loc[:-lg, feat]

    return feed_out


def remove_nans(feed: pd.DataFrame, feature):
    '''
    Remove nan columns from data frame, 
    and returns the non-nan indice
    '''
    feed_out = feed[feature].copy()
    feed_out = feed_out[~np.isnan(feed_out)]
    idx = feed_out.index

    return feed_out, idx


def create_x_y(feed, target, feature, x_nan_zero=True):
    feed_temp = feed.copy().reset_index()
    x = feed_temp[feature]
    y = feed_temp[target]
    y, idx = remove_nans(feed_temp, target)
    x = x.iloc[idx].values

    if x_nan_zero:
        x[np.isnan(x)] = 0

    return x, y


def fit_temporal_filter(estimator, x, y):
    # Estimator has to be an instantiated scikit-learn class.
    # That has at least a fit function.
    estimator.fit(x, y)

    try:
        filter_weights = estimator.coef_
    except AttributeError:
        raise AttributeError(
            "Use a sklearn object that has a .coef_ attribute (i.e. Ridge, SVR)!")

    try:
        filter_intercept = estimator.intercept_
    except AttributeError:
        warnings.warn(
            "Estimator does not contain intercept, replacing with 0!")
        filter_intercept = 0

    return filter_weights, filter_intercept, estimator
