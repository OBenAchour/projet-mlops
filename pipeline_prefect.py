import subprocess
import sys
import argparse

import mlflow
from monitoring import log_to_elasticsearch
import mlflow.sklearn
from prefect import task, flow

# Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Customer_Churn")


# ============================================================
#                         TASKS
# ============================================================


@task(name="install_dependencies", log_prints=True)
def install_dependencies():
    print("📦 Installation des dépendances...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True,
    )
    print("✅ Dépendances installées.")


@task(name="format_code", log_prints=True)
def format_code():
    print("🎨 Formatage du code avec Black...")
    subprocess.run(
        ["black", ".", "--exclude", "/(venv|__pycache__|reports)/"],
        check=True,
    )
    print("✅ Code formaté.")


@task(name="check_code_quality", log_prints=True)
def check_code_quality():
    print("🔍 Vérification de la qualité du code avec Flake8...")
    result = subprocess.run(
        ["flake8", ".", "--exclude", "venv,__pycache__,reports"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode == 0:
        print("✅ Aucun problème de style détecté.")
    else:
        print("⚠️ Des problèmes de style ont été détectés (non bloquant).")


@task(name="check_code_security", log_prints=True)
def check_code_security():
    print("🔒 Vérification de la sécurité avec Bandit...")
    result = subprocess.run(
        ["bandit", "-r", ".", "-x", "./venv,./__pycache__,./reports"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode == 0:
        print("✅ Aucun problème de sécurité détecté.")
    else:
        print("⚠️ Des problèmes de sécurité ont été détectés (non bloquant).")


@task(name="run_tests", log_prints=True)
def run_tests():
    print("🧪 Exécution des tests unitaires avec Pytest...")
    subprocess.run(
        ["pytest", "--junitxml=reports/pytest.xml", "-v"],
        check=False,
    )
    print("✅ Tests passés avec succès.")


@task(name="prepare_data", log_prints=True)
def prepare_data_task():
    print("📥 Préparation des données...")
    from data_loader import prepare_data

    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    print(f"✅ Données préparées : train={X_train.shape}, test={X_test.shape}")
    return X_train, X_test, y_train, y_test


@task(name="train_model", log_prints=True)
def train_model_task(X_train, y_train):
    print("🚀 Entraînement du modèle...")
    from trainer import train_model

    model = train_model(X_train, y_train)
    return model


@task(name="evaluate_model", log_prints=True)
def evaluate_model_task(model, X_test, y_test):
    print("📈 Évaluation du modèle...")
    from evaluator import evaluate_model

    accuracy = evaluate_model(model, X_test, y_test)
    return accuracy


@task(name="save_model", log_prints=True)
def save_model_task(model):
    print("💾 Sauvegarde du modèle...")
    from model_io import save_model

    save_model(model, "classifier.joblib")


@task(name="log_model_mlflow", log_prints=True)
def log_model_mlflow_task(model):
    print("📦 Enregistrement du modèle dans MLflow...")
    mlflow.sklearn.log_model(model, "model")
    print("✅ Modèle enregistré dans MLflow.")


@task(name="load_and_predict", log_prints=True)
def predict_task():
    print("🔮 Chargement du modèle et prédiction...")
    from model_io import load_model
    from data_loader import prepare_data

    model = load_model("classifier.joblib")
    X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")
    predictions = model.predict(X_test[:5])
    print(f"📊 Prédictions (5 premiers) : {predictions}")
    return predictions


@task(name="launch_api", log_prints=True)
def launch_api_task():
    print("🌐 Lancement du serveur API (uvicorn)...")
    subprocess.Popen(
        ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
    )
    print("✅ Serveur lancé sur http://0.0.0.0:8000")


@task(name="clean_artifacts", log_prints=True)
def clean_task():
    import os
    import glob

    print("🧹 Nettoyage des artefacts...")
    patterns = ["__pycache__", "*.pyc", "reports/*.txt", "reports/*.xml"]
    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            if os.path.isfile(f):
                os.remove(f)
                print(f"  Supprimé : {f}")
    print("✅ Nettoyage terminé.")


# ============================================================
#                         FLOWS
# ============================================================


@flow(name="install", log_prints=True)
def install_flow():
    install_dependencies()


@flow(name="code", log_prints=True)
def code_quality_flow():
    format_code()
    check_code_quality()
    check_code_security()


@flow(name="test", log_prints=True)
def test_flow():
    run_tests()


@flow(name="prepare", log_prints=True)
def prepare_flow():
    prepare_data_task()


@flow(name="train", log_prints=True)
def train_flow():
    with mlflow.start_run(run_name="prefect-train"):
        X_train, X_test, y_train, y_test = prepare_data_task()
        model = train_model_task(X_train, y_train)
        save_model_task(model)
        log_model_mlflow_task(model)


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    with mlflow.start_run(run_name="prefect-evaluate"):
        X_train, X_test, y_train, y_test = prepare_data_task()
        model = train_model_task(X_train, y_train)
        evaluate_model_task(model, X_test, y_test)
        log_model_mlflow_task(model)


@flow(name="predict", log_prints=True)
def predict_flow():
    predict_task()


@flow(name="clean", log_prints=True)
def clean_flow():
    clean_task()


@flow(name="api", log_prints=True)
def api_flow():
    launch_api_task()


@flow(name="all", log_prints=True)
def all_flow():
    print("=" * 50)
    print("   Customer Churn - Pipeline ML Complet (Prefect + MLflow)")
    print("=" * 50)

    install_dependencies()
    format_code()
    check_code_quality()
    check_code_security()
    run_tests()

    with mlflow.start_run(run_name="prefect-all"):
        X_train, X_test, y_train, y_test = prepare_data_task()
        model = train_model_task(X_train, y_train)
        evaluate_model_task(model, X_test, y_test)
        save_model_task(model)
        log_model_mlflow_task(model)
        predict_task()

    print("✅ Pipeline complet terminé.")


# ============================================================
#                         MAIN
# ============================================================

FLOWS = {
    "all": all_flow,
    "install": install_flow,
    "code": code_quality_flow,
    "test": test_flow,
    "prepare": prepare_flow,
    "train": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "clean": clean_flow,
    "api": api_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline ML avec Prefect + MLflow")
    parser.add_argument(
        "--flow",
        type=str,
        default="all",
        choices=FLOWS.keys(),
        help="Flow à exécuter",
    )
    args = parser.parse_args()

    selected_flow = FLOWS[args.flow]
    selected_flow()
