from flask import Flask, request, jsonify
from waitress import serve

import config
import logger as logger
from repository import repository_service

from algorithms.vit import vit_train

# flask
app = Flask(__name__)


@app.route('/')
def home():
    return 'Projeto VITA'


@app.route('/train_vit', methods=['GET'])
def train_vit():
    # train_vit
    df = repository_service.get_df_from_csv("X_train")  # loads dataframe

    data = vit_train.train(df)
    print(data)
    return data.to_json(orient='records')


@app.route('/predict_vit', methods=['GET'])
def predict_vit():
    # predict_vit
    df = repository_service.get_df_from_csv("X_train")  # loads dataframe

    data = vit_train.train(df)
    print(data)
    return data.to_json(orient='records')


if __name__ == '__main__':
    logger.log.debug("--- Application starts ---")
    serve(app, host=config.host, port=config.port)  # WSGI server
    logger.log.debug("--- Application ends ---")
