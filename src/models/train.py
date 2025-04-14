import joblib
from sklearn.ensemble import RandomForestRegressor
from ..config.config import DATA_PATH, MODEL_PATH
from ..data_processing.preprocessing import preprocess_data
from ..utils.helpers import save_model, load_data



def train_model():
    df = load_data(DATA_PATH)
    (X_train, X_test, y_train, y_test), encoder, scaler = preprocess_data(df)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump((encoder, scaler), MODEL_PATH.replace("model.pkl", "preprocessors.pkl"))

    score = model.score(X_test, y_test)
    print(f"Model R^2 score: {score:.4f}")

if __name__ == "__main__":
    train_model()
