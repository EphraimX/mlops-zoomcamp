import joblib


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