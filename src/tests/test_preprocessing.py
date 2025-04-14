import pandas as pd
from src.data_processing.preprocessing import preprocess_data

def test_preprocess():
    df = pd.read_csv("data/raw/used_car.csv")
    try:
        (X_train, X_test, y_train, y_test), encoder, scaler = preprocess_data(df)
        assert X_train.shape[0] > 0
        print("✅ Preprocessing test passed.")
    except Exception as e:
        print("❌ Preprocessing test failed:", e)

if __name__ == "__main__":
    test_preprocess()
