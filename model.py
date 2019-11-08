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

def cluster_zillow(data):
    kmeans = KMeans(n_clusters = 3)
    encoder = LabelEncoder()
    scaler = MinMaxScaler()
    train, test = train_test_split(data, random_state = 123)
    kmeans.fit(train[['logerror', 'taxvaluedollarcnt']])
    scaler.fit_transform(train)
    train['cluster_group'] = kmeans.labels_
    print(train)
    print('AVERAGE LOG ERROR BY CLUSTER \n%s' % (train.groupby(kmeans.labels_)['logerror'].mean()))
    sns.scatterplot(x = 'logerror', y = 'taxvaluedollarcnt', data = train, hue=kmeans.labels_, c = 'green')
    plt.show()
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
    print(math.sqrt(MSE))



    


