import numpy as np
import pandas as pd
from scipy.signal import convolve
import warnings


def aggregate_data(feed, feature, target, lags):
    x = feed[feature].reset_index()
    y = feed[target].reset_index()

    y_non_nan = y[~np.isnan(y[target])]
    # In order to circumvent further errors, when values are not present in the data.
    y_idx = y_non_nan.index
    y_idx = y_idx[y_idx > lags]
    y_non_nan = y_non_nan.loc[y_idx]
    x_agg = np.zeros((y_non_nan.shape[0], lags + 1))

    # Aggregate data:
    for n, steps in enumerate(range(-lags, 1)):
        x_agg[:, n] = x.loc[y_idx + steps][feature].values

    # Maybe drop nans from data?

    x_agg[np.isnan(x_agg)] = 0

    return x_agg, x, y_non_nan, y_idx


def fit_temporal_filter(estimator, x, y):
    # Estimator has to be an instantiated scikit-learn class.
    # That has at least a fit function. 
    estimator.fit(x, y)

    try:
        filter_weights = estimator.coef_
    except AttributeError:
        raise AttributeError("Use a sklearn object that has a .coef_ attribute (i.e. Ridge, SVR)!")

    try:
        filter_intercept = estimator.intercept_
    except AttributeError:
        warnings.warn("Estimator does not contain intercept, replacing with 0!")
        filter_intercept = 0
    
    return filter_weights, filter_intercept, estimator