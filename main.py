import sys

import mlflow
import mlflow.sklearn

from data_loader import prepare_data
from trainer import train_model
from evaluator import evaluate_model
from model_io import save_model, load_model

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer_Churn")


def main():

    if len(sys.argv) == 1:

        print("=" * 40)
        print("   Customer Churn - Pipeline ML")
        print("=" * 40)

        with mlflow.start_run():

            X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

            model = train_model(X_train, y_train)

            evaluate_model(model, X_test, y_test)

            mlflow.sklearn.log_model(model, "model")

            save_model(model, "classifier.joblib")

            load_model("classifier.joblib")

            print("✅ Pipeline terminé.")
        return

    step = sys.argv[1]

    if step == "-prepare_data":
        print("📥 Préparation des données...")
        prepare_data("Churn_Modelling.csv")

    elif step == "-train_model":
        print("🚀 Entraînement du modèle...")
        with mlflow.start_run():
            X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
            model = train_model(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")

    elif step == "-evaluate_model":
        print("📈 Évaluation du modèle...")
        model = load_model("classifier.joblib")
        X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        with mlflow.start_run():
            evaluate_model(model, X_test, y_test)

    elif step == "-save_model":
        print("💾 Sauvegarde du modèle...")
        X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
        model = train_model(X_train, y_train)
        save_model(model, "classifier.joblib")

    elif step == "-load_model":
        print("🔄 Chargement du modèle...")
        load_model("classifier.joblib")

    else:
        print("❌ Argument inconnu :", step)
        print("Arguments disponibles :")
        print("-prepare_data")
        print("-train_model")
        print("-evaluate_model")
        print("-save_model")
        print("-load_model")


if __name__ == "__main__":
    main()
