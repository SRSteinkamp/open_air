#%% Import cell
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from scipy.signal import convolve
from polair import polair
DATA_PATH = 'data/'
DATA_BASE = '2'
#% Data Loading and some very basic preprocessing:
df_lanuv = pd.read_csv(f'{DATA_PATH}df_lanuv{DATA_BASE}.csv', index_col='timestamp')
df_openair = pd.read_csv(f'{DATA_PATH}df_openair{DATA_BASE}.csv', index_col='timestamp')

#% Unnamed: 0 can be removed later, data is incorrectly saved.
df_lanuv = df_lanuv.drop(columns=['Unnamed: 0'])
df_openair = df_openair.drop(columns=['Unnamed: 0'])

# Collapse data, so that we can access the different stations
# directly for each timestamp
df_lanuv = df_lanuv.pivot(columns='station')['no2']
df_lanuv = pd.DataFrame(df_lanuv.to_records())
#% df openair - set -1 to NaN!
df_openair[df_openair.values == -1] = np.nan
#% Merge the data. Many nan's in the LANUV data now.
df_joined = pd.merge(df_openair, df_lanuv, how='left', on=['timestamp'])
df_joined['timestamp'] = pd.to_datetime(df_joined['timestamp'])

# Question: is a diff based on timestamps feasible?
# Drop nans from df_joined.
#% 
feeds = np.unique(df_joined['feed'])


#%
feed_idx = 2
sel_feed = feeds[feed_idx]
single_feed = df_joined.query('feed == @sel_feed')

#%%

# Interesting features for a first look:
# r2, temp, hum
# implement an automatic check for usability of r1

# Idea 1 implement multiple temporal filters. For hum, r2, and temp. 
# Weighted average of filters.
# Needed to do: Create a test set. Better ways to aggregate the data. It's a bit complicated.
filter_dictionary = dict()

train = np.arange(106)[:80]
test = np.arange(106)[80:]

for targ in ['CHOR', 'RODE', 'VKCL', 'VKTU']:
    predicted = []
    x_feat = []
    for feat in ['r2', 'hum', 'temp']:
        x_agg, x, y, y_idx = polair.aggregate_data(single_feed, feat, targ, 2)

        temp_filter, temp_intercept, _ = polair.fit_temporal_filter(RidgeCV(), x_agg[train], y[targ].values[train])

        temp = np.convolve(x[feat].values, temp_filter, mode='same')  + temp_intercept
        print(r2_score(y[targ].values[test], temp[y_idx][test]))
        filter_dictionary[feat, targ] = (temp_filter, temp_intercept)
        predicted.append(temp[y_idx])
        x_feat.append(x[feat].values)

    predicted = np.stack(predicted).T
    x_feat = np.stack(x_feat).T
    #%%
    combiner = RidgeCV(fit_intercept=False)
    combiner.fit(predicted[train], y[targ].values[train])

    combined_features = np.sum(predicted[test] * combiner.coef_, 1)
    print(r2_score(y[targ].values[test], combined_features))
#%% Combine filter:
