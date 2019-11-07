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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
import math

warnings.filterwarnings("ignore")



def cluster_zillow(data):
    kmeans = KMeans(n_clusters = 4)
    encoder = LabelEncoder()
    data = data.drop(columns = ['transactiondate', 'propertycountylandusecode'])
    train, test = train_test_split(data, random_state = 123)
    train['taxdelinquencyflag'] = encoder.fit_transform(train[['taxdelinquencyflag']])
    kmeans.fit(train[['logerror', 'taxvaluedollarcnt']])
    print(kmeans.cluster_centers_)
    
    #sns.scatterplot(x = 'logerror', y = 'taxvaluedollarcnt', data = train, hue=kmeans.labels_, c = 'green')
    #sns.scatterplot('latitude', 'longitude' , hue = kmeans.labels_, data = test)
    print(train.info())
    #sns.pairplot(train)
    #plt.show()
    return train, test


def model_zillow(train, test):
    model = LinearRegression()
    x1=train.drop(columns=['logerror'])
    x2=test.drop(columns=['logerror'])
    y1=train[['logerror']]
    y2=test[['logerror']]
    model.fit(x1, y1)
    y_pred = model.predict(y1)
    #y_pred_proba = model.predict_proba(x2)
    MSE = mean_squared_error(y1, y_pred)
    print(math.sqrt(MSE))



    