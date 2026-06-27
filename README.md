# Customer Churn — ML Pipeline & API

Projet MLOps de prédiction du churn client basé sur le dataset `Churn_Modelling.csv`, avec un pipeline orchestré par **Prefect**, une API REST via **FastAPI** et un suivi des expériences via **MLflow**.

---

## Architecture du projet

```
ML_Project/
├── data_loader.py            # Préparation des données (encoding, scaling, split)
├── trainer.py                # Entraînement du modèle + log MLflow params
├── evaluator.py              # Évaluation + log MLflow metrics & artefacts
├── model_io.py               # Sauvegarde / chargement du modèle (joblib)
├── main.py                   # Point d'entrée classique (avec MLflow)
├── app.py                    # API FastAPI (/predict, /retrain)
├── pipeline_prefect.py       # Pipeline Prefect + MLflow (tasks + flows)
├── deploiement_prefect.py    # Déploiements Prefect (scheduling)
├── quality_check.sh          # Script qualité (black, flake8, pylint, bandit, pytest)
├── requirements.txt          # Dépendances Python
├── classifier.joblib         # Modèle entraîné
├── mlflow.db                 # Base de données MLflow (tracking local)
├── Churn_Modelling.csv       # Dataset
├── confusion_matrix.png      # Matrice de confusion générée
├── reports/                  # Rapports qualité
└── tests/
    ├── test_data_loader.py
    ├── test_trainer.py
    ├── test_evaluator.py
    └── test_model_io.py
```

---

## 1. Installation

### 1.1 Prérequis

- Python 3.10+
- WSL / Ubuntu
- Environnement virtuel Python

### 1.2 Mise en place

```bash
cd ML_Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Pipeline classique (sans Prefect)

Exécuter le pipeline complet :

```bash
python main.py
```

Exécuter une étape spécifique :

```bash
python main.py -prepare_data
python main.py -train_model
python main.py -evaluate_model
python main.py -save_model
python main.py -load_model
```

---

## 3. Pipeline Prefect

### 3.1 Description

Le fichier `pipeline_prefect.py` orchestre les étapes du projet sous forme de **tasks** et **flows** Prefect :

| Flow        | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `all`       | Pipeline complet (install → qualité code → tests → ML + MLflow → predict) |
| `install`   | Installation des dépendances depuis requirements.txt                     |
| `code`      | Formatage (black) + qualité (flake8) + sécurité (bandit)                |
| `test`      | Exécution des tests unitaires (pytest)                                   |
| `prepare`   | Préparation des données                                                  |
| `train`     | Préparation + entraînement + sauvegarde + log MLflow                     |
| `evaluate`  | Préparation + entraînement + évaluation + log MLflow                     |
| `predict`   | Chargement du modèle + prédiction sur 5 échantillons                     |
| `clean`     | Nettoyage des artefacts (__pycache__, reports)                           |
| `api`       | Lancement du serveur uvicorn                                             |

### 3.2 Lancement direct

```bash
source venv/bin/activate
python pipeline_prefect.py --flow all
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow code
python pipeline_prefect.py --flow test
```

### 3.3 Déploiement avec scheduling (3 terminaux)

**Terminal 1** — Serveur Prefect (UI sur http://localhost:4200) :

```bash
source venv/bin/activate
prefect server start
```

**Terminal 2** — Worker (exécute les flows) :

```bash
source venv/bin/activate
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
python deploiement_prefect.py
```

**Terminal 3** — Déclenchement manuel :

```bash
source venv/bin/activate
prefect deployment ls
prefect deployment run 'all/ml-pipeline-all'
prefect deployment run 'train/ml-pipeline-train'
prefect deployment run 'evaluate/ml-pipeline-evaluate'
```

Le déploiement `ml-pipeline-all` est programmé pour s'exécuter automatiquement tous les jours à 2h du matin (cron `0 2 * * *`).

---

## 4. API FastAPI

### 4.1 Description

Le fichier `app.py` expose le modèle entraîné comme un service REST avec deux endpoints :

| Endpoint     | Méthode | Description                                         |
|--------------|---------|-----------------------------------------------------|
| `/`          | GET     | Vérification que l'API tourne                        |
| `/predict`   | POST    | Prédiction du churn pour un client                   |
| `/retrain`   | POST    | Réentraînement du modèle avec nouveaux hyperparamètres |

### 4.2 Lancement

```bash
source venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Documentation Swagger : http://127.0.0.1:8000/docs

### 4.3 Test de /predict

Depuis Swagger ou avec `curl` :

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619,
    "Gender": 0,
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

Réponse attendue :

```json
{
  "prediction": 0,
  "churn": false,
  "probability": {
    "no_churn": 0.89,
    "churn": 0.11
  }
}
```

#### Features attendues

| Champ            | Type  | Description                          |
|------------------|-------|--------------------------------------|
| CreditScore      | float | Score de crédit du client             |
| Gender           | int   | 0 = Female, 1 = Male                 |
| Age              | float | Âge du client                         |
| Tenure           | int   | Nombre d'années en tant que client    |
| Balance          | float | Solde du compte                       |
| NumOfProducts    | int   | Nombre de produits bancaires          |
| HasCrCard        | int   | Possède une carte de crédit (0/1)     |
| IsActiveMember   | int   | Membre actif (0/1)                    |
| EstimatedSalary  | float | Salaire estimé                        |

### 4.4 Test de /retrain

```bash
curl -X POST http://127.0.0.1:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{
    "n_estimators": 200,
    "random_state": 42
  }'
```

Réponse attendue :

```json
{
  "message": "Modèle réentraîné avec succès",
  "accuracy": 86.35,
  "params": {
    "n_estimators": 200,
    "random_state": 42
  }
}
```

---

## 5. MLflow — Suivi des expériences

### 5.1 Description

MLflow est intégré dans le projet pour suivre les expériences ML. Chaque run enregistre automatiquement :

| Type          | Éléments loggés                                          |
|---------------|----------------------------------------------------------|
| Paramètres    | `model`, `n_estimators`, `random_state`                  |
| Métriques     | `accuracy`, `precision`, `recall`, `f1_score`            |
| Artefacts     | `confusion_matrix.png`, modèle sklearn                   |

### 5.2 Configuration

Le tracking utilise une base SQLite locale (`mlflow.db`), ce qui évite de devoir lancer un serveur MLflow séparé pendant l'exécution du pipeline.

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn")
```

### 5.3 Lancement via Prefect + MLflow

```bash
source venv/bin/activate
python pipeline_prefect.py --flow train      # log params + modèle
python pipeline_prefect.py --flow evaluate   # log params + métriques + modèle
python pipeline_prefect.py --flow all        # pipeline complet avec tracking
```

### 5.4 Consulter les résultats dans l'UI MLflow

Après avoir exécuté un ou plusieurs runs :

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

Puis ouvrir dans le navigateur : http://127.0.0.1:5000

L'interface permet de comparer les runs, visualiser les métriques, télécharger les artefacts et consulter les modèles enregistrés.

### 5.5 Note pour Python 3.14

Si MLflow échoue avec `ImportError: cannot import name 'Traversable' from 'importlib.abc'`, appliquer ce fix :

```bash
grep -rl "from importlib.abc import Traversable" venv/lib/python3.14/site-packages/mlflow/ \
  | xargs sed -i 's/from importlib.abc import Traversable/from importlib.resources.abc import Traversable/g'
```

---

## 6. Qualité du code

Le script `quality_check.sh` exécute 5 vérifications :

```bash
bash quality_check.sh
```

| Étape  | Outil   | Rôle                                  | Rapport              |
|--------|---------|---------------------------------------|----------------------|
| 1/5    | Black   | Formatage automatique du code         | reports/black.txt    |
| 2/5    | Flake8  | Vérification du style PEP8           | reports/flake8.txt   |
| 3/5    | Pylint  | Analyse qualité approfondie           | reports/pylint.txt   |
| 4/5    | Bandit  | Détection de failles de sécurité      | reports/bandit.txt   |
| 5/5    | Pytest  | Exécution des tests unitaires         | reports/pytest.txt   |

---

## 7. Tests unitaires

```bash
pytest -v
```

Les tests couvrent les 4 modules principaux :

| Fichier                 | Module testé    |
|-------------------------|-----------------|
| test_data_loader.py     | data_loader.py  |
| test_trainer.py         | trainer.py      |
| test_evaluator.py       | evaluator.py    |
| test_model_io.py        | model_io.py     |

---

## 8. Dépendances

```
pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, openpyxl
black, flake8, pylint, bandit, pytest
prefect, fastapi, uvicorn, mlflow
```
