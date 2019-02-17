# %% Import cell
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.signal import convolve
from polair import polair
# %%
date = '19-02-17-17'

DATA_PATH = 'data/'
# % Data Loading and some very basic preprocessing:
df_lanuv = pd.read_csv(
    f'{DATA_PATH}{date}_df_lanuv.csv', index_col='timestamp')
df_openair = pd.read_csv(
    f'{DATA_PATH}{date}_df_openair.csv', index_col='timestamp')

# Collapse data, so that we can access the different stations
# directly for each timestamp
df_lanuv = df_lanuv.pivot(columns='station')['no2']
df_lanuv = pd.DataFrame(df_lanuv.to_records())
# % df openair - set -1 to NaN!
df_openair[df_openair.values == -1] = np.nan
# % Merge the data. Many nan's in the LANUV data now.
df_joined = pd.merge(df_openair, df_lanuv, how='left', on=['timestamp'])
df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])

# Question: is a diff based on timestamps feasible?
# Drop nans from df_joined.
# %%
feeds = np.unique(df_joined['feed'])


# %
feed_idx = 0
sel_feed = feeds[feed_idx]
single_feed = df_joined.query('feed == @sel_feed').copy()
single_feed = single_feed.reset_index()
lags = 2


filter_dictionary = dict()

features = ['r2', 'hum', 'temp']
single_feed_aug = polair.create_lags(single_feed, features, lags)

for targ in ['CHOR']:  # , 'RODE', 'VKCL', 'VKTU']:
    predicted = []

    # Train and test set:
    y_glob, idx = polair.remove_nans(single_feed_aug, targ)

    train = idx[: np.int(len(idx) * 0.8)]
    test = idx[np.int(len(idx) * 0.8):]
    train_idx = np.arange(train.shape[0])
    test_idx = np.arange(train.shape[0], train.shape[0] + test.shape[0])
    for feat in ['r2', 'hum', 'temp']:

        uni_features = [
            f'{feat}_lg{ii}' for ii in reversed(range(1, lags + 1))]
        uni_features.append(feat)
        x, y = polair.create_x_y(single_feed_aug, targ, uni_features)

        temp_filter, temp_intercept, _ = polair.fit_temporal_filter(
            RidgeCV(), x[train_idx], y.values[train_idx])

        temp = np.convolve(single_feed[feat].values, temp_filter[::-1],
                           mode='same') + temp_intercept
        temp[np.isnan(temp)] = 0
        print(r2_score(y.values[test_idx], temp[test]))
        filter_dictionary[feat, targ] = (temp_filter, temp_intercept)
        predicted.append(temp)

    predicted = np.stack(predicted).T
    combiner = RidgeCV(fit_intercept=False)
    combiner.fit(predicted[train], y_glob[train])
    combined_features = np.sum(predicted[test] * combiner.coef_, 1)
    print(r2_score(y[test], combined_features))
# %% Combine filter:
