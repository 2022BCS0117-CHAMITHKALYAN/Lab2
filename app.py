from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Wine Quality Prediction API")

MODEL_PATH = "model.pkl"

# Load model
model = joblib.load(MODEL_PATH)

# ----------------------------
# Input schema (Swagger UI)
# ----------------------------

class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# ----------------------------
# Prediction Endpoint
# ----------------------------

@app.post("/predict")
def predict(features: WineFeatures):
    # Convert input to model format (same order as training)
    x = np.array([[
        features.fixed_acidity,
        features.volatile_acidity,
        features.citric_acid,
        features.residual_sugar,
        features.chlorides,
        features.free_sulfur_dioxide,
        features.total_sulfur_dioxide,
        features.density,
        features.pH,
        features.sulphates,
        features.alcohol
    ]])

    # Predict using model
    pred = model.predict(x)[0]

    return {
        "name": "BOLLI CHAMITH KALYAN",
        "roll_no": "2022BCS0117",
        "wine_quality": int(pred)
    }
