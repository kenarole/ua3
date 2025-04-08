import requests

url = "http://127.0.0.1:8000/predict"
sample_data = {
    "features": [5.1, 3.5, 1.4, 0.2]  # adapte selon ton modèle
}

response = requests.post(url, json=sample_data)
print("Réponse de l'API :", response.json())
