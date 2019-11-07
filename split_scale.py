import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def split_my_data(X):
    train, test = train_test_split(X, train_size =.80, random_state = 123)
    return train, test


def standard_scaler(X):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(X)
    scaled_X = pd.DataFrame(scaler.transform(X),columns=X.columns.values).set_index([X.index.values])
    return scaled_X






