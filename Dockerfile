FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching).
# Use the slimmer production requirements for the runtime image.
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt \
    && rm -rf /root/.cache/pip

# Application code and the trained model artifact.
COPY app.py data_loader.py trainer.py evaluator.py model_io.py monitoring.py pipeline_prefect.py ./
COPY classifier.joblib Churn_Modelling.csv ./

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness check against the FastAPI app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/docs').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
