from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("decision_tree_model.pkl")

# Charger le scaler
scaler = joblib.load("scaler.pkl")


# Initialiser l'API
app = FastAPI(title="API Prédiction avec Decision Tree")

# Schéma des données attendues
class InputData(BaseModel):
    features: list  # Exemple : [3.2, 1.5, 0.8, ...]

# @app.post("/predict")
# def predict(data: InputData):
#     X = np.array(data.features).reshape(1, -1)
#     prediction = model.predict(X)[0]
#     return {"prediction": int(prediction)}


@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    return {"prediction": int(prediction)}

