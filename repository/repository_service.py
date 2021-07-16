import pickle
import os
import pandas as pd
import config
import logger as logger

# path where is trained models and loaders
path_dataset = config.path + os.path.sep + "datasets" + os.path.sep
path_model = config.path + os.path.sep + "trained" + os.path.sep + "models" + os.path.sep
path_normalizer = config.path + os.path.sep + "trained" + os.path.sep + "normalizers" + os.path.sep

# create path is not exist
if not os.path.exists(path_dataset):
    os.makedirs(path_dataset)
if not os.path.exists(path_model):
    os.makedirs(path_model)
if not os.path.exists(path_normalizer):
    os.makedirs(path_normalizer)


def get_df_from_csv(name):
    """ returns filename as dataframe

    :param name: filename, example: data
    :return: dataframe in case of success or None in case of exception

    """

    try:
        filename = path_dataset + name + '.csv'
        df = pd.read_csv(filename, sep=';', engine='python')
        logger.log.info(filename + " loaded")
        return df
    except Exception as e:
        logger.log.info("RepositoryService get_df exception " + str(e))
        return None

