import joblib

from flask import Flask, request, jsonify


app = Flask('duration-prediction')

with open('web-service/model.bin', 'rb') as f_in:
    dv, model = joblib.load(f_in)


def prepare_features(ride):
    features = {}
    features['PU DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration' : pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)