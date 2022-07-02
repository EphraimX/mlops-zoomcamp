import joblib


with open('model.bin', 'rb') as f_in:
    dv, model = joblib.load(f_in)


def prepare_features():
    features = {}
    features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds 