

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# 1. PROJECT PATHS
# Works both locally on Windows and inside Docker
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

if (BASE_DIR / "artifacts").exists():
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
else:
    ARTIFACTS_DIR = BASE_DIR.parent / "artifacts"

STOCKOUT_DIR = ARTIFACTS_DIR / "stockout"
ANOMALY_DIR = ARTIFACTS_DIR / "anomaly"


# ============================================================
# 2. LOAD TRAINED MODELS AND PREPROCESSORS
# ============================================================

try:
    stockout_model = tf.keras.models.load_model(
        STOCKOUT_DIR / "best_stockout_mlp.keras"
    )

    stockout_preprocessor = joblib.load(
        STOCKOUT_DIR / "stockout_preprocessor.pkl"
    )

    anomaly_model = tf.keras.models.load_model(
        ANOMALY_DIR / "best_anomaly_autoencoder.keras"
    )

    anomaly_scaler = joblib.load(
        ANOMALY_DIR / "anomaly_scaler.pkl"
    )

    anomaly_threshold = float(
        np.load(ANOMALY_DIR / "best_anomaly_threshold.npy")
    )

except Exception as e:
    raise RuntimeError(
        f"Could not load model artifacts. "
        f"Artifacts directory: {ARTIFACTS_DIR}. "
        f"Error: {e}"
    ) from e


# ============================================================
# 3. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI-Powered Supply Chain API",
    description=(
        "An API for stockout prediction using a Multi-Layer "
        "Perceptron (MLP) and anomaly detection using an Autoencoder."
    ),
    version="1.0.0"
)


# ============================================================
# 4. INPUT DATA SCHEMAS
# ============================================================

class StockoutInput(BaseModel):
    category: str
    demand: float = Field(ge=0)
    inventory_level: float = Field(ge=0)
    orders: float = Field(ge=0)
    price: float = Field(ge=0)
    promotion: str
    lead_time_days: float = Field(ge=0)
    supplier_rating: float
    transport_cost: float = Field(ge=0)
    warehouse_capacity: float = Field(ge=0)


class AnomalyInput(BaseModel):
    demand: float = Field(ge=0)
    inventory_level: float = Field(ge=0)
    orders: float = Field(ge=0)
    price: float = Field(ge=0)
    lead_time_days: float = Field(ge=0)
    supplier_rating: float
    transport_cost: float = Field(ge=0)
    warehouse_capacity: float = Field(ge=0)


# ============================================================
# 5. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Welcome to the AI-Powered Supply Chain API",
        "status": "API is running",
        "available_tasks": [
            "Stockout Prediction",
            "Anomaly Detection"
        ],
        "documentation": "/docs"
    }


# ============================================================
# 6. STOCKOUT PREDICTION
# Model: Multi-Layer Perceptron (MLP)
# ============================================================

@app.post("/predict/stockout")
def predict_stockout(data: StockoutInput):
    try:
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Match the training feature order when available
        if hasattr(stockout_preprocessor, "feature_names_in_"):
            feature_names = list(
                stockout_preprocessor.feature_names_in_
            )
            input_df = input_df[feature_names]

        # Apply the same preprocessing used during training
        processed_input = stockout_preprocessor.transform(input_df)

        # Convert sparse matrix if necessary
        if hasattr(processed_input, "toarray"):
            processed_input = processed_input.toarray()

        processed_input = np.asarray(
            processed_input,
            dtype=np.float32
        )

        # Make prediction
        model_output = stockout_model.predict(
            processed_input,
            verbose=0
        )

        probability = float(
            np.asarray(model_output).reshape(-1)[0]
        )

        # Classify using a 0.5 probability threshold
        prediction = int(probability >= 0.5)

        if prediction == 1:
            result = "Stockout risk detected"
            recommendation = (
                "Review current inventory and expected demand. "
                "Consider replenishment if appropriate."
            )
        else:
            result = "Low stockout risk predicted"
            recommendation = (
                "Continue monitoring inventory and demand."
            )

        return {
            "task": "Stockout Prediction",
            "stockout_prediction": prediction,
            "result": result,
            "stockout_probability_percent": round(
                probability * 100, 2
            ),
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Stockout prediction failed: {str(e)}"
        ) from e


# ============================================================
# 7. ANOMALY DETECTION
# Model: Autoencoder
# ============================================================

@app.post("/predict/anomaly")
def predict_anomaly(data: AnomalyInput):
    try:
        # Keep feature order consistent with training
        feature_names = [
            "demand",
            "inventory_level",
            "orders",
            "price",
            "lead_time_days",
            "supplier_rating",
            "transport_cost",
            "warehouse_capacity"
        ]

        input_df = pd.DataFrame(
            [data.model_dump()]
        )[feature_names]

        # Scale the input
        scaled_input = anomaly_scaler.transform(input_df)

        scaled_input = np.asarray(
            scaled_input,
            dtype=np.float32
        )

        # Reconstruct input using the Autoencoder
        reconstructed = anomaly_model.predict(
            scaled_input,
            verbose=0
        )

        # Calculate reconstruction error
        reconstruction_error = float(
            np.mean(
                np.square(scaled_input - reconstructed)
            )
        )

        # Compare error with the saved threshold
        prediction = int(
            reconstruction_error > anomaly_threshold
        )

        if prediction == 1:
            result = "Anomalous record detected"
            recommendation = (
                "Review this record and investigate unusual "
                "demand, inventory, orders, or transport values."
            )
        else:
            result = "Record appears normal"
            recommendation = (
                "No anomaly detected by the model. "
                "Continue routine monitoring."
            )

        return {
            "task": "Anomaly Detection",
            "is_anomaly": prediction,
            "result": result,
            "reconstruction_error": round(
                reconstruction_error, 6
            ),
            "threshold": round(
                anomaly_threshold, 6
            ),
            "recommendation": recommendation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        ) from e
