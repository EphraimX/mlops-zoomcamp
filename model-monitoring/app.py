# from crypt import methods
import os
import pickle
import re
import requests

from pymongo.mongo_client import MongoClient



from flask import Flask, request, jsonify

MODEL_FILE = os.getenv('MODEL_FILE', 'lin_reg.bin')
MONGODB_ADDRESS = os.getenv('MONGODB_ADDRESS', 'mongodb://127.0.0.0.1:27017')
EVIDENTLY_SERVICE_ADDRESS = os.getenv('EVIDENTY_SERVICE_', 'http://127.0.0.1:5000')

with open(MODEL_FILE, 'rb') as f_in:
    dv, model = pickle.load(f_in)


app = Flask("duration")
mongo_client = MongoClient(MONGODB_ADDRESS)
db = mongo_client.get_database('prediction_service')
collection = db.get_collection('data')


@app.route('/predict', methods=['POST'])
def predict():
    record = request.get_json()
    record['PU_DO'] = '%s_%s' % (record['PULocationID'], record['DOLocationID'])
    X = dv.transform([record])
    y_pred = model.predict(X)

    prediction= y_pred

    result = {
        'duration' : float(y_pred)
    }

    save_to_db(record, prediction)
    save_to_evidently_service(record, prediction)

    return jsonify(result)


def save_to_db(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    collection.insert_one(rec)


def save_to_evidently_service(record, prediction):
    rec = record.copy()
    rec['prediction'] = prediction
    requests.post(f'{EVIDENTLY_SERVICE_ADDRESS}/iterate/taxi', json=[rec])
