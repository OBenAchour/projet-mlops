# Commandes — Comment runner chaque étape

---

## 1. Environnement

```bash
wsl
cd /mnt/c/Users/ousse/ML_Project
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Pipeline classique

```bash
python main.py                    # pipeline complet
python main.py -prepare_data
python main.py -train_model
python main.py -evaluate_model
python main.py -save_model
python main.py -load_model
```

---

## 3. Pipeline Prefect

### Lancement direct

```bash
python pipeline_prefect.py --flow all
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow evaluate
python pipeline_prefect.py --flow predict
python pipeline_prefect.py --flow code
python pipeline_prefect.py --flow test
python pipeline_prefect.py --flow install
python pipeline_prefect.py --flow clean
```

### Deploiement avec scheduling

```bash
# Terminal 1 — Serveur Prefect
prefect server start

# Terminal 2 — Worker
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
python deploiement_prefect.py

# Terminal 3 — Declencheur
prefect deployment ls
prefect deployment run 'all/ml-pipeline-all'
prefect deployment run 'train/ml-pipeline-train'
prefect deployment run 'evaluate/ml-pipeline-evaluate'
```

---

## 4. API FastAPI + Dashboard

```bash
# Lancer l'API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Dashboard : http://localhost:8000/dashboard
# Swagger :   http://localhost:8000/docs
```

### Tester /predict avec curl

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

### Tester /retrain avec curl

```bash
curl -X POST http://127.0.0.1:8000/retrain \
  -H "Content-Type: application/json" \
  -d '{"n_estimators": 200, "random_state": 42}'
```

---

## 5. MLflow

```bash
# Lancer le serveur en background
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 &

# Executer un pipeline avec tracking
python pipeline_prefect.py --flow train
python pipeline_prefect.py --flow evaluate

# UI : http://localhost:5000

# Fix Python 3.14
grep -rl "from importlib.abc import Traversable" \
  venv/lib/python3.14/site-packages/mlflow/ \
  | xargs sed -i 's/from importlib.abc import Traversable/from importlib.resources.abc import Traversable/g'
```

---

## 6. Docker

```bash
# Construire l'image
docker build -t oussema_benachour_3asi_mlops .

# Lancer le conteneur
docker run -d -p 8000:8000 --name ml-api oussema_benachour_3asi_mlops

# Verifier
docker ps

# Arreter
docker stop ml-api && docker rm ml-api

# Se connecter a Docker Hub
docker login -u obenachour

# Taguer et pousser
docker tag oussema_benachour_3asi_mlops obenachour/oussema_benachour_3asi_mlops:latest
docker push obenachour/oussema_benachour_3asi_mlops:latest
```

### Via Makefile

```bash
make build
make run
make stop
make login
make push
make train
make evaluate
make all
make test
make quality
```

---

## 7. Monitoring (Elasticsearch + Kibana)

```bash
# Lancer la stack
docker-compose up -d

# Verifier Elasticsearch
curl http://localhost:9200

# Envoyer des metriques de test
python3 -c "
from monitoring import log_to_elasticsearch
log_to_elasticsearch(
    'test-run',
    {'model': 'RandomForestClassifier', 'n_estimators': 100},
    {'accuracy': 0.856, 'precision': 0.782, 'recall': 0.424, 'f1_score': 0.55}
)
print('OK')
"

# Verifier dans Elasticsearch
curl http://localhost:9200/mlflow-metrics/_search?pretty
curl http://localhost:9200/mlflow-metrics/_count?pretty

# Arreter la stack
docker-compose down
```

### Kibana

```
http://localhost:5601
  -> Stack Management -> Index Patterns -> Create "mlflow-metrics"
  -> Time field : timestamp
  -> Discover pour visualiser
```

---

## 8. Qualite du code

```bash
# Script complet
bash quality_check.sh

# Individuellement
black . --exclude '/(venv|__pycache__|reports)/'
flake8 . --exclude venv,__pycache__,reports
pylint main.py data_loader.py trainer.py evaluator.py model_io.py
bandit -r . -x ./venv,./__pycache__,./reports
pytest -v
```

---

## 9. Tout lancer (ordre recommande)

```bash
# 1. Activer l'environnement
cd /mnt/c/Users/ousse/ML_Project
source venv/bin/activate

# 2. Lancer Elasticsearch + Kibana
docker-compose up -d

# 3. Lancer MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 &

# 4. Lancer Prefect
prefect server start &

# 5. Executer le pipeline
python pipeline_prefect.py --flow all

# 6. Lancer l'API + Dashboard
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 7. Ouvrir le dashboard
# http://localhost:8000/dashboard
```

---

## URLs de toutes les interfaces

| Service         | URL                              |
|-----------------|----------------------------------|
| Dashboard       | http://localhost:8000/dashboard   |
| Swagger API     | http://localhost:8000/docs        |
| Prefect         | http://localhost:4200             |
| MLflow          | http://localhost:5000             |
| Kibana          | http://localhost:5601             |
| Elasticsearch   | http://localhost:9200             |
