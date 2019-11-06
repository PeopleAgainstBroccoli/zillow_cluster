import acquire
import numpy as np


def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing/rows
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': pct_missing})
    return cols_missing

def nulls_by_row(df):
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing



def prep_zillow(data):
    data.info()
    zillow_data = data
    assumed_zero = ['fireplacecnt', 'garagecarcnt', 'hashottubeorspa', 'poolcnt']
    for a in assumed_zero:
        zillow_data[a].apply(lambda x: 0 if x == np.nan() else x)
    subset = zillow_data[['latitude', 'longitude', 'taxvaluedollarcnt', 'logerror', 'fips']]
    subset['fips'] = encoder.fit_transform(data['fips'])
    subset['poor_people'] = subset['taxvaluedollarcnt'] < 1000000
    subset['log_me'] = (subset['logerror'] < .9) & (subset['logerror'] > -.9)
    subset = subset.loc[subset.poor_people, :]
    subset = subset.loc[subset.log_me, :]
    return subset


