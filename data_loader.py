import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(filepath="Churn_Modelling.csv"):
    df = pd.read_csv(filepath)

    encoder = LabelEncoder()
    df["Gender"] = encoder.fit_transform(df["Gender"])

    df.drop(columns=["Surname", "Geography"], inplace=True)

    X = df.drop(columns=["Exited", "RowNumber", "CustomerId"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
