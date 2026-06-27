from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)

    print("✅ Modèle entraîné avec succès.")
    return model
