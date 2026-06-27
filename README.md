# Customer Churn — MLOps Pipeline

Projet MLOps complet de prédiction du churn client, intégrant orchestration, API REST, tracking d'expériences, conteneurisation, monitoring et dashboard de contrôle.

---

## Architecture du projet

```
ML_Project/
├── app.py                    # API FastAPI + Dashboard de contrôle
├── data_loader.py            # Préparation des données
├── trainer.py                # Entraînement du modèle + log MLflow
├── evaluator.py              # Évaluation + log MLflow
├── model_io.py               # Sauvegarde / chargement du modèle
├── monitoring.py             # Envoi des métriques vers Elasticsearch
├── main.py                   # Point d'entrée classique
├── pipeline_prefect.py       # Pipeline Prefect + MLflow
├── deploiement_prefect.py    # Déploiements Prefect schedulés
├── quality_check.sh          # Script qualité de code
├── Dockerfile                # Image Docker pour l'API
├── docker-compose.yml        # Stack Elasticsearch + Kibana
├── Makefile                  # Automatisation des tâches
├── requirements.txt          # Dépendances Python
├── classifier.joblib         # Modèle entraîné
├── mlflow.db                 # Base MLflow
├── Churn_Modelling.csv       # Dataset
├── confusion_matrix.png      # Matrice de confusion
├── reports/                  # Rapports qualité
└── tests/                    # Tests unitaires
```

---

## Interfaces disponibles

| Service         | URL                              | Description                          |
|-----------------|----------------------------------|--------------------------------------|
| Dashboard       | http://localhost:8000/dashboard   | Contrôle complet du projet           |
| Swagger API     | http://localhost:8000/docs        | Documentation interactive de l'API   |
| Prefect         | http://localhost:4200             | Orchestration des flows              |
| MLflow          | http://localhost:5000             | Suivi des expériences ML             |
| Kibana          | http://localhost:5601             | Visualisation des métriques          |
| Elasticsearch   | http://localhost:9200             | Stockage des logs                    |

---

## Installation rapide

```bash
cd ML_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Pipeline Prefect

| Flow        | Description                                              |
|-------------|----------------------------------------------------------|
| `all`       | Pipeline complet avec tracking MLflow                    |
| `install`   | Installation des dépendances                             |
| `code`      | Formatage + qualité + sécurité du code                  |
| `test`      | Tests unitaires                                          |
| `prepare`   | Préparation des données                                  |
| `train`     | Entraînement + sauvegarde + log MLflow                   |
| `evaluate`  | Évaluation + métriques + log MLflow                      |
| `predict`   | Prédiction sur échantillons                              |
| `clean`     | Nettoyage des artefacts                                  |
| `api`       | Lancement du serveur API                                 |

---

## API FastAPI

| Endpoint     | Méthode | Description                              |
|--------------|---------|------------------------------------------|
| `/dashboard` | GET     | Dashboard de contrôle complet            |
| `/predict`   | POST    | Prédiction du churn                      |
| `/retrain`   | POST    | Réentraînement du modèle                 |
| `/api/run-flow` | POST | Lancer un flow Prefect                |
| `/api/services-status` | GET | Statut de tous les services      |
| `/api/metrics` | GET   | Dernières métriques Elasticsearch        |

---

## MLflow — Tracking

| Type          | Éléments loggés                                  |
|---------------|--------------------------------------------------|
| Paramètres    | `model`, `n_estimators`, `random_state`          |
| Métriques     | `accuracy`, `precision`, `recall`, `f1_score`    |
| Artefacts     | `confusion_matrix.png`, modèle sklearn           |

---

## Docker

```bash
make build    # Construire l'image
make run      # Lancer le conteneur
make push     # Pousser sur Docker Hub
make stop     # Arrêter le conteneur
```

Image Docker Hub : `obenachour/oussema_benachour_3asi_mlops`

---

## Monitoring

Elasticsearch stocke les métriques envoyées par `monitoring.py`. Kibana les visualise via l'index `mlflow-metrics`.

---

## Qualité du code

| Outil   | Rôle                   |
|---------|------------------------|
| Black   | Formatage automatique  |
| Flake8  | Style PEP8             |
| Pylint  | Qualité approfondie    |
| Bandit  | Sécurité               |
| Pytest  | Tests unitaires        |

---

## Dépendances

```
pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, openpyxl
black, flake8, pylint, bandit, pytest
prefect, fastapi, uvicorn, mlflow, elasticsearch, httpx
```
