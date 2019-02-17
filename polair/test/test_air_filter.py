import context
import polair
import numpy as np
import pandas as pd
from nose.tools import raises
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#%%
def create_feed():
    y = np.arange(20).astype('float')
    y[[1, 5, 10]] = np.nan
    x = np.arange(20).astype('float')
    feed = pd.DataFrame.from_dict({'y': y, 'x': x})

    return feed
#%%

def test_aggregate_data_x_shape():
    # Small test for dimensions of x
    lags = 2
    test_data = create_feed()
    x, _, _, y_idx = polair.aggregate_data(test_data, 'x', 'y', lags)

    assert np.all(x.shape == (y_idx.shape[0], lags + 1))


def test_aggregate_data_y_shape():
    # Small test for dimensions of y
    lags = 2
    test_data = create_feed()
    x, _, y, _ = polair.aggregate_data(test_data, 'x', 'y', lags)

    assert np.all(x.shape[0] == y.shape[0])


def test_aggregate_data_y_idx():
    # Small test if nans are remove from y
    lags = 2
    test_data = create_feed()
    _, _, _, y = polair.aggregate_data(test_data, 'x', 'y', lags)

    assert y.shape[0] ==  (np.sum(np.isnan(test_data['y']) == 0)) - 2


def test_aggregate_data_x_ident():
    # X in should be x out
    lags = 2
    test_data = create_feed()
    _, x, _, _ = polair.aggregate_data(test_data, 'x', 'y', lags)

    assert np.allclose(x['x'].values, test_data['x'].values)


@raises(AttributeError)
def test_temporal_filter_coef():
    test_data = create_feed()
    test_data = test_data.fillna(0)
    estimator = RandomForestRegressor(n_estimators=1)
    polair.fit_temporal_filter(estimator, test_data['x'].values.reshape(-1, 1), test_data['y'].values)


def test_temporal_filter_filter_size():
    test_data = create_feed()
    lags = 2
    x_agg, _, y, _ = polair.aggregate_data(test_data, 'x', 'y', lags)
    estimator = LinearRegression()
    fw, _, _ = polair.fit_temporal_filter(estimator, x_agg, y)

    assert fw.shape[0] == (lags + 1)


def test_temporal_filter_intercept_size():
    test_data = create_feed()
    lags = 2
    x_agg, _, y, _ = polair.aggregate_data(test_data, 'x', 'y', lags)
    estimator = LinearRegression(fit_intercept=False)
    _, intercept, _ = polair.fit_temporal_filter(estimator, x_agg, y)
    
    assert intercept == 0