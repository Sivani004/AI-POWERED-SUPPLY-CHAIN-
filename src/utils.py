
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


# --------------------------------------------------
# 2. LOAD PROCESSED DATA
# --------------------------------------------------

def load_npz(folder, filename):
    file_path = ARTIFACTS_DIR / folder / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {file_path}\n"
            "Please run preprocessing.py first."
        )

    with np.load(file_path, allow_pickle=False) as data:
        return {
            key: data[key]
            for key in data.files
        }


def load_demand_data():
    return load_npz(
        "demand",
        "demand_processed.npz"
    )


def load_stockout_data():
    return load_npz(
        "stockout",
        "stockout_processed.npz"
    )


def load_anomaly_data():
    return load_npz(
        "anomaly",
        "anomaly_processed.npz"
    )


# --------------------------------------------------
# 3. DEMAND FORECASTING METRICS
# --------------------------------------------------

def evaluate_demand(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


# --------------------------------------------------
# 4. STOCKOUT PREDICTION METRICS
# --------------------------------------------------

def evaluate_stockout(
    y_true,
    y_prob,
    threshold=0.5
):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_prob = np.asarray(y_prob).reshape(-1)

    y_pred = (y_prob >= threshold).astype(int)

    if len(y_true) != len(y_prob):
        raise ValueError(
            "Stockout labels and predictions must have "
            "the same number of observations."
        )

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# --------------------------------------------------
# 5. ANOMALY DETECTION HELPERS
# --------------------------------------------------

def calculate_reconstruction_errors(model, X):
    X = np.asarray(X)

    reconstructed = model.predict(
        X,
        verbose=0
    )

    errors = np.mean(
        np.square(X - reconstructed),
        axis=1
    )

    return errors


def evaluate_anomaly(
    y_true,
    errors,
    threshold
):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    errors = np.asarray(errors).reshape(-1)

    if len(y_true) != len(errors):
        raise ValueError(
            "Anomaly labels and reconstruction errors "
            "must have the same length."
        )

    y_pred = (errors >= threshold).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, errors)
        )
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# --------------------------------------------------
# 6. MLFLOW LOGGING HELPERS
# --------------------------------------------------

def log_task_params(task_name, params):
    for key, value in params.items():

        if isinstance(value, (list, tuple)):
            value = ",".join(
                str(item) for item in value
            )

        mlflow.log_param(
            f"{task_name}_{key}",
            value
        )


def log_task_metrics(metrics, prefix=""):
    for key, value in metrics.items():

        value = float(value)

        if np.isfinite(value):
            mlflow.log_metric(
                f"{prefix}{key}",
                value
            )


# --------------------------------------------------
# 7. TRAINING CALLBACKS
# --------------------------------------------------

def make_callbacks(
    patience=5,
    monitor="val_loss",
    mode="min"
):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=True,
        verbose=0,
    )

    return [early_stopping]


# --------------------------------------------------
# 8. SAVE TRAINED MODEL
# --------------------------------------------------

def save_model(model, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(str(file_path))

    return file_path


