import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def save_model(model, path):
    import joblib
    joblib.dump(model, path)

def describe_dataset(df):
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.describe(include="all"))