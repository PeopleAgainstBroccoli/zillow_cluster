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


def min_max_scaler(X):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(X)
    scaled_X = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    return scaled_X
def standard_scaler(X):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
    scaled_X = pd.DataFrame(scaler.transform(X),columns=X.columns.values).set_index([X.index.values])
    return scaled_X


def results_train(logit, y_pred, y_pred_proba, x_train, y_train):
    print('\n\n<<<<<<<<<<<<<<<<|RESULTS|>>>>>>>>>>>>>>>\n')
    try:
        print('Accuracy of Logistic Regression classifier on training set: {:.2f} \n'
              .format(logit.score(x_train, y_train)))
    except:
        pass
    try:
        print('Accuracy of Logistic Regression classifier on test set: {:.2f} \n'
              .format(logit.score(x_test, y_test)))
    except:
        pass
    try:
        print('Coefficient: \n', logit.coef_)
        print('Intercept: \n', logit.intercept_)
    except:
        pass
    print('')
    print('-----------|CONFUSION_MATRIX|------------')
    try:
        print(confusion_matrix(y_train, y_pred))
    except:
        print('<<|UNKOWN|>>')
    print('-----------------|REPORT|-----------------')
    try:
        print(classification_report(y_train, y_pred))
    except:
        print('<<|UNKNOWN|>>')
    print('----------------------------------------')










def return_xy(train):
    x1 = train[['bedroomcnt', 'poolcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', \
                'taxdelinquencyflag', 'taxdelinquencyyear']]
    #x1 = train[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxdelinquencyflag', 'cluster']]
    #x2 = test[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxdelinquencyflag']]
    y1=train[['logerror']]
    return x1, y1



def cluster_zillow(data):
    kmeans = KMeans(n_clusters = 3)
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    train, test = train_test_split(data, random_state = 123)
    kmeans.fit(train[['taxvaluedollarcnt']])
    scaler.fit_transform(train)
    train['cluster_group'] = kmeans.labels_
    print('AVERAGE LOG ERROR BY CLUSTER \n%s' % (train.groupby(kmeans.labels_)['logerror'].mean()))
    train['cluster'] = kmeans.labels_
    sns.scatterplot(x = 'logerror', y = 'taxvaluedollarcnt', hue = kmeans.labels_, data = train)
    plt.show()
    #g = sns.FacetGrid(train, col = 'cluster')
    #g = g.map(plt.scatter, 'latitude', 'longitude', alpha = .5)
    #sns.scatterplot('latitude', 'longitude', data = train, hue=kmeans.labels_, c = 'green')
    
    #plt.show()
    return train, test

def average_log_error_cluster(data):
    print(data)
    kmeans = KMeans(n_clusters = 3)
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    train, test = train_test_split(data, random_state = 123)
    kmeans.fit(train[['taxvaluedollarcnt']])
    scaler.fit_transform(train)
    sns.scatterplot('logerror', 'taxvaluedollarcnt', hue = kmeans.labels_, data = data)
    plt.show()
    print('AVERAGE LOG ERROR BY CLUSTER \n%s' % (train.groupby(kmeans.labels_)['logerror'].mean()))





def baseline_model_zillow(train):
    model = LinearRegression()
    x1=train.drop(columns=['logerror'])
    y1=train[['logerror']]
    model.fit(x1, y1)
    y_pred = model.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)


def model_zillow_linear(x1, y1):
    model = LinearRegression()
    model.fit(x1, y1)
    y_pred = model.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)


def model_zillow_tree(x1, y1):
    tree = DecisionTreeRegressor(max_depth = 8, random_state = 123)
    tree.fit(x1, y1)
    y_pred = tree.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)
    


def model_zillow_forest(x1, y1):
    forest = RandomForestRegressor(max_depth = 8, random_state = 123).fit(x1, y1)
    y_pred = forest.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)


def model_zillow_forest_test(x1, y1, x2, y2):
    forest = RandomForestRegressor(max_depth = 5, random_state = 123).fit(x1, y1)
    y_pred = forest.predict(x2)
    MSE = mean_squared_error(y2, y_pred)
    return math.sqrt(MSE)

    

def mean_log_error(data, y_train):
    y_pred = (y_train['logerror'] == y_train['logerror'].sum()) / len(y_train)
    MSE = mean_squared_error(y_train, y_pred)
    return math.sqrt(MSE)
