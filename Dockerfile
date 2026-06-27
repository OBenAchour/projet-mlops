FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY data_loader.py .
COPY trainer.py .
COPY evaluator.py .
COPY model_io.py .
COPY classifier.joblib .
COPY Churn_Modelling.csv .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
