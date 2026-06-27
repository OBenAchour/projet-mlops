from elasticsearch import Elasticsearch
from datetime import datetime


es = Elasticsearch(["http://localhost:9200"])


def log_to_elasticsearch(run_name, params, metrics):
    doc = {
        "run_name": run_name,
        "timestamp": datetime.utcnow().isoformat(),
        "params": params,
        "metrics": metrics,
    }

    es.index(index="mlflow-metrics", document=doc)
    print(f"📊 Métriques envoyées à Elasticsearch : {metrics}")
