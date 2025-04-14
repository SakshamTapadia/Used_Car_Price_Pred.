import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from numpy import hstack

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Create a copy to avoid the SettingWithCopyWarning
    df = df.copy()

    # Drop missing values
    df = df.dropna()

    # Clean 'AskPrice' column: Remove currency symbols and commas, then convert to float
    df['AskPrice'] = df['AskPrice'].replace({'₹': '', ',': ''}, regex=True).astype(float)

    # Separate features (X) and target variable (y)
    X = df.drop("AskPrice", axis=1)
    y = df["AskPrice"]

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    # Initialize OneHotEncoder and StandardScaler
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()

    # Encode categorical features and scale numerical features
    X_cat = encoder.fit_transform(X[categorical_cols]) if len(categorical_cols) > 0 else None
    X_num = scaler.fit_transform(X[numerical_cols]) if len(numerical_cols) > 0 else None

    # Concatenate the processed categorical and numerical features
    if X_cat is not None and X_num is not None:  # If both categorical and numerical columns exist
        X_processed = hstack([X_num, X_cat])
    elif X_cat is not None:  # If only categorical columns exist
        X_processed = X_cat
    elif X_num is not None:  # If only numerical columns exist
        X_processed = X_num
    else:
        X_processed = None  # In case both are empty (shouldn't happen)

    # Split the data into training and testing sets
    return train_test_split(X_processed, y, test_size=0.2, random_state=42), encoder, scaler
