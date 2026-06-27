FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip

COPY app.py .
COPY data_loader.py .
COPY trainer.py .
COPY evaluator.py .
COPY model_io.py .
COPY monitoring.py .
COPY pipeline_prefect.py .
COPY classifier.joblib .
COPY Churn_Modelling.csv .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
