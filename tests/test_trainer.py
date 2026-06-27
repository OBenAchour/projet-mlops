from data_loader import prepare_data
from trainer import train_model


def test_train_model():

    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

    model = train_model(X_train, y_train)

    assert model is not None

    assert hasattr(model, "predict")
