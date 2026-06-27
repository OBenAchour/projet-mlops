from data_loader import prepare_data
from trainer import train_model
from evaluator import evaluate_model


def test_evaluate_model():

    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

    model = train_model(X_train, y_train)

    accuracy = evaluate_model(model, X_test, y_test)

    assert accuracy >= 0
    assert accuracy <= 1
