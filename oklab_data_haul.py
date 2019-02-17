# %%
import polair as pol
import pandas as pd
import numpy as np
import datetime
# %%
host_name, database_name = pol.get_db_from_csv('data_base_info.csv')
# %%
LAST_N_DAYS = 15
AGGREGATE = 'median'
TIME_AGGREGATE = 20  # This is in minutes!!!
# The date of the query!
date = datetime.datetime.now().strftime("%y-%m-%d-%H")


lanuv_query = ("SELECT station, NO2 as no2 "
               "FROM lanuv_f2 "
               f"WHERE time >= now() - {LAST_N_DAYS}d "
               "AND time <= now() ")


openair_query = (f"SELECT {AGGREGATE}(hum) as hum, "
                 f"{AGGREGATE}(r1) as r1, "
                 f"{AGGREGATE}(r2) as r2, "
                 f"{AGGREGATE}(temp) as temp, "
                 f"{AGGREGATE}(rssi) as rssi "
                 "FROM all_openair "
                 f"WHERE time >= now() - {LAST_N_DAYS}d "
                 "AND time <= now() "
                 f"GROUP BY feed, time({TIME_AGGREGATE}m) fill(-1)")

# %% Querying LANUV data
lanuv_dict = pol.query_influx(
    host_name, database_name, lanuv_query)

# Basic preprocessing as done by codefor muenster
df_lanuv = lanuv_dict['lanuv_f2'].rename_axis('timestamp').reset_index()
# Clean out left over minutes or seconds
df_lanuv = df_lanuv[df_lanuv.timestamp.dt.minute == 0]
df_lanuv = df_lanuv[df_lanuv.timestamp.dt.second == 0]

df_lanuv = df_lanuv.assign(timestamp=pd.to_datetime(df_lanuv.timestamp.astype(np.int64)
                                                    // 10 ** 6, unit='ms', utc=True))
# %%
df_lanuv.to_csv(f'data/{date}_df_lanuv.csv', index=False)


# %% Query openair
openair_dict = pol.query_influx(
    host_name, database_name, openair_query)
# %%
openair_dict_clean = {k[1][0][1]: openair_dict[k] for k in openair_dict.keys()}
df_openair = pd.DataFrame()

for feed in list(openair_dict_clean.keys()):
    df_feed = (pd.DataFrame.from_dict(openair_dict_clean[feed])
               .assign(feed=feed).rename_axis('timestamp').reset_index())
    df_openair = df_openair.append(df_feed)

df_openair.to_csv(f'data/{date}_df_openair.csv', index=False)

# %%
