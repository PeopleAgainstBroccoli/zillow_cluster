import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
warnings.filterwarnings("ignore")
import math

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


def min_max_scaler(X):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X)
    scaled_X = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    return scaled_X
def standard_scaler(X):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
    scaled_X = pd.DataFrame(scaler.transform(X),columns=X.columns.values).set_index([X.index.values])
    return scaled_X

def prep_zillow(data):
    encoder = LabelEncoder()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    mode_imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
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
    
    impute_mode = ['regionidcity', 'censustractandblock']
    impute_mean = [ 'calculatedfinishedsquarefeet', 'lotsizesquarefeet', 'regionidzip',\
                    'yearbuilt', 'structuretaxvaluedollarcnt', 'censustractandblock']


    data['taxdelinquencyflag'] = encoder.fit_transform(data[['taxdelinquencyflag']])
    for i in impute_mean:

        zillow_data[i] = imputer.fit_transform(zillow_data[[i]])
        
    for i in impute_mode:
        zillow_data[i] = mode_imputer.fit_transform(zillow_data[[i]])
        
    #impute regionidcity, calculatedfinishedsquarefeet, lotsizesquarefeet, regionidzip, yearbuilt, structuretaxvaluedollarcnt, censustractandblock
    zillow_data = zillow_data.drop(columns = ['finishedsquarefeet15', 'finishedsquarefeet13', 'buildingclasstypeid', \
                                'storytypeid', 'pooltypeid2', 'pooltypeid10', 'pooltypeid7','basementsqft', \
                                'typeconstructiontypeid', 'fireplaceflag', 'calculatedbathnbr', 'airconditioningtypeid', 'architecturalstyletypeid', \
                                              'finishedfloor1squarefeet', 'taxamount','rawcensustractandblock','finishedsquarefeet50', 'finishedsquarefeet12', 'finishedsquarefeet6', \
                                              'garagetotalsqft', 'fullbathcnt', 'censustractandblock', 'hashottuborspa', 'heatingorsystemtypeid', 'poolsizesum', \
                                              'propertyzoningdesc', 'regionidneighborhood', 'threequarterbathnbr', \
                                              'yardbuildingsqft26', 'structuretaxvaluedollarcnt', 'id', 'yardbuildingsqft17', \
                                              'unitcnt', 'decktypeid', 'landtaxvaluedollarcnt','assessmentyear', 'log_me', 'poor_people',\
                                              'buildingqualitytypeid', 'numberofstories', 'transactiondate', 'propertycountylandusecode'])
    



    #subset = zillow_data[['latitude', 'longitude', 'taxvaluedollarcnt', 'logerror', 'fips']]
    zillow_data['fips'] = encoder.fit_transform(zillow_data['fips'])
    zillow_data = zillow_data.dropna()
    zillow_data = min_max_scaler(zillow_data)
    return zillow_data


