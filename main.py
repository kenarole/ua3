from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class FeatureInput(BaseModel):
    features: List[List[float]]  # <== liste de vecteurs

@app.post("/predict")
def predict(data: FeatureInput):
    predictions = []
    for row in data.features:
        # 🔁 ton modèle ici
        pred = 1 if row[0] > 5 else 0
        predictions.append(pred)
    return {"predictions": predictions}

