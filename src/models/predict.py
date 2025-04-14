import joblib
import pandas as pd
from src.config.config import MODEL_PATH

def predict(input_dict):
    model = joblib.load(MODEL_PATH)
    encoder, scaler = joblib.load(MODEL_PATH.replace("model.pkl", "preprocessors.pkl"))

    df = pd.DataFrame([input_dict])
    X_cat = encoder.transform(df.select_dtypes(include=["object"]))
    X_num = scaler.transform(df.select_dtypes(exclude=["object"]))

    from numpy import hstack
    X_processed = hstack([X_num, X_cat])

    return model.predict(X_processed)[0]
