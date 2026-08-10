"""
API FastAPI pour la détection de fraude bancaire.

Charge le modèle exporté par Airflow (model.pkl) et expose un endpoint /predict
qui reçoit une transaction et retourne une prédiction (fraude ou non) avec sa
probabilité. Expose aussi /metrics pour le monitoring Prometheus.
"""

import json
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(
    title="API de détection de fraude bancaire",
    description="Prédit si une transaction par carte bancaire est frauduleuse.",
    version="1.0.0",
)

MODEL_PATH = "/app/model/model.pkl"
FEATURE_COLUMNS_PATH = "/app/model/feature_columns.json"

# On charge le modèle et la liste des colonnes une seule fois, au démarrage de l'API
# (pas à chaque requête, ce serait beaucoup trop lent)
model = None
feature_columns = None

# --- Métriques Prometheus ---
# Compteur : combien de prédictions au total, ventilées par résultat (fraude/normale)
# -> permet de suivre le TAUX de fraude détecté dans le temps sur un graphique Grafana
predictions_total = Counter(
    "fraud_predictions_total",
    "Nombre total de prédictions effectuées, par résultat",
    ["label"]
)

# Histogramme : distribution du temps de réponse de /predict
# -> permet de vérifier le respect de l'exigence "< 200ms" du cahier des charges
prediction_latency = Histogram(
    "fraud_prediction_latency_seconds",
    "Temps de traitement d'une prédiction (secondes)"
)

# Histogramme : distribution des probabilités de fraude retournées
# -> utile pour repérer une DÉRIVE : si le modèle se met à retourner des probabilités
# très différentes de d'habitude, c'est un signal que les données ont changé (data drift)
prediction_probability = Histogram(
    "fraud_prediction_probability",
    "Distribution des probabilités de fraude prédites",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


@app.on_event("startup")
def charger_modele():
    global model, feature_columns
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    print(f"Modèle chargé avec succès ({len(feature_columns)} colonnes attendues)")


@app.get("/metrics")
def metrics():
    """Endpoint que Prometheus vient interroger périodiquement pour collecter les métriques."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class Transaction(BaseModel):
    """
    Une transaction par carte bancaire à analyser.

    V1 à V28 sont des composantes issues d'une transformation PCA (anonymisation
    des données originales) ; Time et Amount sont les seules colonnes en clair.
    """
    Time: float = Field(..., description="Secondes écoulées depuis la première transaction du dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Montant de la transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36,
                "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31,
                "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
                "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02, "Amount": 149.62
            }
        }


class PredictionResponse(BaseModel):
    prediction: int  # 0 = transaction normale, 1 = fraude
    label: str
    probabilite_fraude: float


@app.get("/")
def racine():
    return {"message": "API de détection de fraude - voir /docs pour la documentation interactive"}


@app.get("/health")
def health():
    """Vérifie que l'API tourne et que le modèle est bien chargé."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "modele_charge": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Reçoit une transaction et retourne une prédiction de fraude."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    debut = time.time()

    # On convertit la transaction en DataFrame avec les colonnes dans le MÊME ordre
    # que celui utilisé à l'entraînement (crucial pour la fiabilité de la prédiction)
    donnees = pd.DataFrame([transaction.dict()])
    donnees = donnees[feature_columns]

    prediction = int(model.predict(donnees)[0])
    probabilite = float(model.predict_proba(donnees)[0][1])  # probabilité de la classe "fraude"
    label = "fraude" if prediction == 1 else "normale"

    # Enregistrement des métriques pour Prometheus/Grafana
    predictions_total.labels(label=label).inc()
    prediction_probability.observe(probabilite)
    prediction_latency.observe(time.time() - debut)

    return PredictionResponse(
        prediction=prediction,
        label=label,
        probabilite_fraude=round(probabilite, 4),
    )