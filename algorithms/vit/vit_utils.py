import numpy as np
import pandas as pd
from sklearn import preprocessing


def remove_null_columns(df):
    """
    remove columns null

    :param df: dataframe to process
    :return: dataframe without null columns
    """

    df = df.dropna()
    return df


def scaled(df):
    """
    normaliza os dados entre 0 e 1

    :param df: dataframe para processar
    :return: dados entre 0 e 1
    """
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    return x
