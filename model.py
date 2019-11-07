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
warnings.filterwarnings("ignore")


def model_zillow(data):
    kmeans = KMeans(n_clusters = 4)
    encoder = LabelEncoder()
    data = data.drop(columns = ['transactiondate', 'propertycountylandusecode'])
    train, test = train_test_split(data, random_state = 123)
    train['taxdelinquencyflag'] = encoder.fit_transform(train[['taxdelinquencyflag']])
    kmeans.fit(train[['logerror', 'taxvaluedollarcnt']])
    print(kmeans.cluster_centers_)
    sns.scatterplot(x = 'logerror', y = 'taxvaluedollarcnt', data = train, hue=kmeans.labels_, c = 'green')
    plt.show()




