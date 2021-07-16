import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from sklearn.pipeline import make_pipeline

from algorithms.vit.vit_utils import remove_null_columns, scaled


def preprocessing(df):
    """
    pré processamento dos dados

    :param df: dataframe
    :return: dataframe 'limpo'

    """
    df = remove_null_columns(df)
    df = scaled(df)
    return df


def train(df):
    """
    utiliza os dados para treinamento e teste do modelo e retorna o modelo treinado
    :param df: dataframe com os dados
    :return modelo treinado
    """

    df2 = df.copy()
    print(df2)
    return df2



