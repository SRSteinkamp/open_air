import context
import polair
import numpy as np
import pandas as pd
from nose.tools import raises
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# %%


def create_feed():
    y = np.arange(20).astype('float')
    y[[1, 5, 10]] = np.nan
    x = np.arange(20).astype('float')
    z = x + 1
    feed = pd.DataFrame.from_dict({'y': y, 'x': x, 'z': z})

    return feed


def test_create_lags_element_1():
    test_data = create_feed()
    features = ['x']
    lags = 1
    lag_data = polair.create_lags(test_data, features, lags)

    assert 'x_lg1' in list(lag_data.columns)


def test_create_lags_element_2():
    test_data = create_feed()
    features = ['x', 'z']
    lags = 1
    lag_data = polair.create_lags(test_data, features, lags)

    assert 'x_lg1' in list(
        lag_data.columns) and 'z_lg1' in list(lag_data.columns)


def test_create_lags_element_2_lags():
    test_data = create_feed()
    features = ['x']
    lags = 2
    lag_data = polair.create_lags(test_data, features, lags)

    assert 'x_lg1' in list(
        lag_data.columns) and 'x_lg2' in list(lag_data.columns)


def test_create_lags_data_move_0():
    test_data = create_feed()
    features = ['x']
    lags = 1
    lag_data = polair.create_lags(test_data, features, lags)

    assert lag_data['x'][0] == lag_data['x_lg1'][1]


def test_create_lags_data_move_10():
    test_data = create_feed()
    features = ['x']
    lags = 1
    lag_data = polair.create_lags(test_data, features, lags)

    assert lag_data['x'][10] == lag_data['x_lg1'][11]


@raises(AttributeError)
def test_temporal_filter_coef():
    test_data = create_feed()
    test_data = test_data.fillna(0)
    estimator = RandomForestRegressor(n_estimators=1)
    polair.fit_temporal_filter(
        estimator, test_data['x'].values.reshape(-1, 1), test_data['y'].values)


def test_temporal_filter_filter_size():
    test_data = create_feed()
    lags = 2
    test_data = polair.create_lags(test_data, ['x', 'z'], lags)
    x, y = polair.create_x_y(test_data, 'y', ['x_lg1', 'x_lg2'])
    estimator = LinearRegression()
    fw, _, _ = polair.fit_temporal_filter(estimator, x, y)

    assert fw.shape[0] == (lags)


def test_temporal_filter_intercept_size():
    test_data = create_feed()
    x, y = polair.create_x_y(test_data, 'y', 'x')
    estimator = LinearRegression(fit_intercept=False)
    _, intercept, _ = polair.fit_temporal_filter(
        estimator, x.reshape(-1, 1), y)

    assert intercept == 0
