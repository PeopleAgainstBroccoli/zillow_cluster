import acquire
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


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
    encoder = LabelEncoder()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data.info()
    data['poor_people'] = data['taxvaluedollarcnt'] < 1000000
    data['log_me'] = (data['logerror'] < 1) & (data['logerror'] > -1)
    data = data.loc[data.poor_people, :]
    data = data.loc[data.log_me, :]
    zillow_data = data
    zillow_data['taxdelinquencyflag'] = zillow_data['taxdelinquencyflag'].fillna('N')
    assumed_zero = ['fireplacecnt', 'garagecarcnt', 'poolcnt', 'taxdelinquencyyear']
    for a in assumed_zero:
        zillow_data[a] = zillow_data[a].fillna(0)
    zillow_data = zillow_data.drop(columns = ['finishedsquarefeet15', 'finishedsquarefeet13', 'buildingclasstypeid', \
                                'storytypeid', 'pooltypeid2', 'pooltypeid10', 'pooltypeid7','basementsqft', \
                                'typeconstructiontypeid', 'fireplaceflag', 'calculatedbathnbr', 'airconditioningtypeid', 'architecturalstyletypeid', \
                                              'finishedfloor1squarefeet', 'finishedsquarefeet50', 'finishedsquarefeet12', 'finishedsquarefeet6', \
                                              'garagetotalsqft', 'hashottuborspa', 'heatingorsystemtypeid', 'poolsizesum', \
                                              'propertyzoningdesc', 'regionidneighborhood', 'threequarterbathnbr', \
                                              'yardbuildingsqft26', 'yardbuildingsqft17', 'unitcnt', \
                                                  'buildingqualitytypeid', 'decktypeid', 'numberofstories'])
    

    #subset = zillow_data[['latitude', 'longitude', 'taxvaluedollarcnt', 'logerror', 'fips']]
    zillow_data['fips'] = encoder.fit_transform(zillow_data['fips'])
    zillow_data
    return zillow_data