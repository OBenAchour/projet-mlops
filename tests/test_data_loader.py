from data_loader import prepare_data


def test_prepare_data():

    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

    assert len(y_train) > 0
    assert len(y_test) > 0

    assert X_train.shape[1] == X_test.shape[1]
