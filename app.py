from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

from data_loader import prepare_data
from trainer import train_model
from model_io import save_model

# ============================================================
#                     Configuration
# ============================================================

MODEL_PATH = "classifier.joblib"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API REST pour prédire le churn client",
    version="1.0.0",
)

# Chargement du modèle au démarrage
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé depuis {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"⚠️ Impossible de charger le modèle : {e}")


# ============================================================
#                     Schémas Pydantic
# ============================================================


class CustomerInput(BaseModel):
    CreditScore: float
    Gender: int  # 0 = Female, 1 = Male
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int  # 0 ou 1
    IsActiveMember: int  # 0 ou 1
    EstimatedSalary: float

    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 619,
                "Gender": 0,
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
            }
        }


class RetrainInput(BaseModel):
    n_estimators: int = 100
    random_state: int = 42

    class Config:
        json_schema_extra = {
            "example": {
                "n_estimators": 200,
                "random_state": 42,
            }
        }


# ============================================================
#                     Routes
# ============================================================


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "status": "running"}


@app.post("/predict")
def predict(customer: CustomerInput):
    global model

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Le modèle n'est pas chargé. Entraînez-le d'abord via /retrain.",
        )

    try:
        features = np.array(
            [
                [
                    customer.CreditScore,
                    customer.Gender,
                    customer.Age,
                    customer.Tenure,
                    customer.Balance,
                    customer.NumOfProducts,
                    customer.HasCrCard,
                    customer.IsActiveMember,
                    customer.EstimatedSalary,
                ]
            ]
        )

        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return {
            "prediction": int(prediction[0]),
            "churn": bool(prediction[0] == 1),
            "probability": {
                "no_churn": round(float(probability[0][0]), 4),
                "churn": round(float(probability[0][1]), 4),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def retrain(params: RetrainInput):
    global model

    try:
        X_train, X_test, y_train, y_test = prepare_data("Churn_Modelling.csv")

        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            random_state=params.random_state,
        )
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        save_model(model, MODEL_PATH)

        return {
            "message": "Modèle réentraîné avec succès",
            "accuracy": round(accuracy * 100, 2),
            "params": {
                "n_estimators": params.n_estimators,
                "random_state": params.random_state,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
