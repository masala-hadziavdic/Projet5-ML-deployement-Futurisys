from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import xgboost as xgb

app = FastAPI(title="API XGBoost Model", description="API de prédiction via un modèle XGBoost", version="1.0")

MODEL_PATH = "model_xgb.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    proba = None
    try:
        proba = model.predict_proba(X).tolist()
    except Exception:
        pass
    return {
        "input": data.features,
        "prediction": prediction.tolist(),
        "probabilities": proba
    }

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API XGBoost 🚀"}
