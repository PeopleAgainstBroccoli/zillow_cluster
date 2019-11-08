import dbtools as db
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
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
    x1 = train[['bedroomcnt', 'poolcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', \
                'taxdelinquencyflag', 'taxdelinquencyyear']]
    x2 = test[['bedroomcnt', 'poolcnt', 'taxvaluedollarcnt', 'calculatedfinishedsquarefeet', 'bathroomcnt', \
                'taxdelinquencyflag', 'taxdelinquencyyear']]
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
    print(train)
    print('AVERAGE LOG ERROR BY CLUSTER \n%s' % (train.groupby(kmeans.labels_)['logerror'].mean()))
    train['cluster'] = kmeans.labels_
    print(train)
    #g = sns.FacetGrid(train, col = 'cluster')
    #g = g.map(plt.scatter, 'latitude', 'longitude', alpha = .5)
    sns.scatterplot('latitude', 'longitude', data = train, hue=kmeans.labels_, c = 'green')
    
    plt.show()
    return train, test



#sns.countplot(x = 'logerror', data = train, hue = train.groupby(kmeans.labels_)['logerror'].mean())
#sns.scatterplot('latitude', 'longitude' , hue = kmeans.labels_, data = train)
#print(train.info())


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
    print(math.sqrt(MSE))



def model_zillow_logistic(train, test):
    x1=train.drop(columns=['logerror'])
    x2=test.drop(columns=['logerror'])
    y1=train[['logerror']]
    y2=test[['logerror']]
    logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga').fit(x1, y1)
    y_pred = logit.predict(x1)
    y_pred_proba = logit.predict_proba(x1)
    print(y_pred)




def model_zillow_linear(train, test):
    model = LinearRegression()
    x1, x2, y1, y2 = return_xy(train, test)
    model.fit(x1, y1)
    y_pred = model.predict(x1)
    MSE = mean_squared_error(y1, y_pred)
    print(math.sqrt(MSE))


def model_zillow_tree(train, test):
    x1, x2, y1, y2 = return_xy(train, test)
    tree = DecisionTreeRegressor(max_depth = 6, random_state = 123)
    tree.fit(x1, y1)
    y_pred = tree.predict(x1)
    #y_pred_proba = tree.predict_proba(x1)
    results_train(tree, y_pred, '!', x1, y1)
    


