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

# %%
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
feed_idx = 2
sel_feed = feeds[feed_idx]
single_feed = df_joined.query('feed == @sel_feed')


filter_dictionary = dict()

features = ['r2', 'hum', 'temp']
single_feed_aug = polair.create_lags(single_feed, features)

for targ in ['CHOR', 'RODE', 'VKCL', 'VKTU']:
    predicted = []

    x_feat = []

    for feat in ['r2', 'hum', 'temp']:

        temp_filter, temp_intercept, _ = polair.fit_temporal_filter(
            RidgeCV(), x_agg[train], y[targ].values[train])

        temp = np.convolve(x[feat].values, temp_filter,
                           mode='same') + temp_intercept
        print(r2_score(y[targ].values[test], temp[y_idx][test]))
        filter_dictionary[feat, targ] = (temp_filter, temp_intercept)
        predicted.append(temp[y_idx])
        x_feat.append(x[feat].values)

    predicted = np.stack(predicted).T
    x_feat = np.stack(x_feat).T
    # %%
    combiner = RidgeCV(fit_intercept=False)
    combiner.fit(predicted[train], y[targ].values[train])

    combined_features = np.sum(predicted[test] * combiner.coef_, 1)
    print(r2_score(y[targ].values[test], combined_features))
# %% Combine filter:
