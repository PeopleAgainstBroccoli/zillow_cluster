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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings("ignore")
import math


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


def return_xy(train, test):
    #x1 = train[['bedroomcnt', 'poolcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', \
     #           'taxdelinquencyflag', 'taxdelinquencyyear']]
    #x2 = test[['bedroomcnt', 'poolcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', \
     #           'taxdelinquencyflag', 'taxdelinquencyyear']]
    x1 = train[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxdelinquencyflag']]
    x2 = test[['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxdelinquencyflag']]
    y1=train[['logerror']]
    y2=test[['logerror']]
    return x1, x2, y1, y2




def cluster_zillow(data):
    kmeans = KMeans(n_clusters = 6)
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    train, test = train_test_split(data, random_state = 123)
    kmeans.fit(train[['taxvaluedollarcnt']])
    scaler.fit_transform(train)
    train['cluster_group'] = kmeans.labels_
    print('AVERAGE LOG ERROR BY CLUSTER \n%s' % (train.groupby(kmeans.labels_)['logerror'].mean()))
    train['cluster'] = kmeans.labels_
    sns.scatterplot('latitude', 'longitude', data = train, hue=kmeans.labels_, c = 'green')
    return train, test

def baseline_model_zillow(train, test):
    model = LinearRegression()
    x1=train.drop(columns=['logerror'])
    x2=test.drop(columns=['logerror'])
    y1=train[['logerror']]
    y2=test[['logerror']]
    model.fit(x1, y1)
    y_pred = model.predict(x1)
    #y_pred_proba = model.predict_proba(x2)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)


def model_zillow_linear(train, test):
    model = LinearRegression()
    x1, x2, y1, y2 = return_xy(train, test)
    model.fit(x1, y1)
    y_pred = model.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)


def model_zillow_tree(train, test):
    x1, x2, y1, y2 = return_xy(train, test)
    tree = DecisionTreeRegressor(max_depth = 6, random_state = 123)
    tree.fit(x1, y1)
    y_pred = tree.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)
    


def model_zillow_forest(train, test):
    x1, x2, y1, y2 = return_xy(train, test)
    forest = RandomForestRegressor(max_depth = 10, random_state = 123).fit(x1, y1)
    y_pred = forest.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    return math.sqrt(MSE)

