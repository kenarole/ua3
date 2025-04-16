from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Chargement du modèle entraîné
model = joblib.load("fraud_detection_pipeline.pkl")

app = FastAPI(title="Projet UA3 : Déploiement de Modèles de Machine Learning dans le Cloud",
            description="API pour la détection de fraude sur des transactions financières utilisant un modèle Decision Tree.",
            version="1.0.0")

# Schéma d'entrée
class Transaction(BaseModel):
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float

# Prétraitement
def preprocess_transaction(row):
    row["transactionType"] = row["nameOrig"][0] + row["nameDest"][0]
    row["net_sender"] = row["oldbalanceOrg"] - row["newbalanceOrig"]
    row["net_receiver"] = row["newbalanceDest"] - row["oldbalanceDest"]

    return {
        "amount": row["amount"],
        "oldbalanceOrg": row["oldbalanceOrg"],
        "newbalanceOrig": row["newbalanceOrig"],
        "oldbalanceDest": row["oldbalanceDest"],
        "newbalanceDest": row["newbalanceDest"],
        "net_sender": row["net_sender"],
        "net_receiver": row["net_receiver"],
        "transactionType": row["transactionType"],
        "type": row["type"]
    }

# Endpoint de prédiction
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    row = transaction.dict()
    row_processed = preprocess_transaction(row)

    # Convertir en DataFrame avec ordre des colonnes
    input_df = pd.DataFrame([row_processed])[[
        'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest',
        'net_sender', 'net_receiver',
        'transactionType', 'type'
    ]]

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    message = "Cette transaction est frauduleuse 🚨" if prediction == 1 else "Transaction légitime ✅"

    return {
        "isFraud": int(prediction),
        "fraudProbability": round(float(proba), 4),
        "message": message
    }

# Accueil
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de détection de fraude utilisant un modèle Decision Tree !"}
