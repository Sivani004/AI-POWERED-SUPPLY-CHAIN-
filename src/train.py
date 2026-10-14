

from pathlib import Path
import joblib
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from utils import (
    load_demand_data,
    load_stockout_data,
    load_anomaly_data,
    evaluate_demand,
    evaluate_stockout,
    calculate_reconstruction_errors,
    evaluate_anomaly,
    log_task_params,
    log_task_metrics,
    make_callbacks,
    save_model,
)

# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

DEMAND_DIR = ARTIFACTS_DIR / "demand"
STOCKOUT_DIR = ARTIFACTS_DIR / "stockout"
ANOMALY_DIR = ARTIFACTS_DIR / "anomaly"

for folder in (DEMAND_DIR, STOCKOUT_DIR, ANOMALY_DIR):
    folder.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. HYPERPARAMETER CONFIGURATIONS
# ============================================================

# Demand: select the configuration with the lowest validation MAE.
DEMAND_CONFIGS = [
    {
        "lstm_units": 32, "dense_units": 16, "dropout": 0.2,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "lstm_units": 64, "dense_units": 32, "dropout": 0.2,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "lstm_units": 64, "dense_units": 32, "dropout": 0.3,
        "learning_rate": 0.0005, "epochs": 40, "batch_size": 32
    },
    {
        "lstm_units": 128, "dense_units": 64, "dropout": 0.2,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "lstm_units": 128, "dense_units": 64, "dropout": 0.3,
        "learning_rate": 0.0005, "epochs": 40, "batch_size": 64
    },
]

# Stockout: select the configuration with the highest validation recall.
STOCKOUT_CONFIGS = [
    {
        "layers": [32], "dropout": 0.1,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "layers": [64, 32], "dropout": 0.2,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "layers": [64, 32], "dropout": 0.3,
        "learning_rate": 0.0005, "epochs": 40, "batch_size": 32
    },
    {
        "layers": [128, 64], "dropout": 0.2,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "layers": [128, 64, 32], "dropout": 0.3,
        "learning_rate": 0.0005, "epochs": 40, "batch_size": 64
    },
]

# Anomaly detection: select the configuration with the highest validation F1.
ANOMALY_CONFIGS = [
    {
        "encoder_units": 32, "bottleneck_units": 8, "dropout": 0.1,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "encoder_units": 64, "bottleneck_units": 16, "dropout": 0.1,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "encoder_units": 64, "bottleneck_units": 8, "dropout": 0.2,
        "learning_rate": 0.0005, "epochs": 40, "batch_size": 32
    },
    {
        "encoder_units": 128, "bottleneck_units": 16, "dropout": 0.2,
        "learning_rate": 0.001, "epochs": 30, "batch_size": 32
    },
    {
        "encoder_units": 128, "bottleneck_units": 8, "dropout": 0.3,
        "learning_rate": 0.0005, "epochs": 40, "batch_size": 64
    },
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_class_weights(y):
    """Calculate weights for imbalanced binary classes."""
    y = np.asarray(y).reshape(-1).astype(int)
    classes = np.unique(y)

    if len(classes) < 2:
        raise ValueError(
            "Stockout training split must contain both classes."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )

    return {
        int(label): float(weight)
        for label, weight in zip(classes, weights)
    }


def mlflow_config_values(config):
    """Convert list parameters to strings for MLflow."""
    return {
        key: ",".join(map(str, value))
        if isinstance(value, (list, tuple))
        else value
        for key, value in config.items()
    }


def make_lstm(config, input_shape):
    """Build the demand forecasting LSTM."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(config["lstm_units"]),
        Dropout(config["dropout"]),
        Dense(config["dense_units"], activation="relu"),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=config["learning_rate"]
        ),
        loss="mse",
        metrics=["mae"],
    )

    return model


def make_mlp(config, input_shape):
    """Build the stockout classification MLP."""
    model = Sequential()
    model.add(Input(shape=input_shape))

    for units in config["layers"]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(config["dropout"]))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=Adam(
            learning_rate=config["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def make_autoencoder(config, input_dim):
    """Build the anomaly detection autoencoder."""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(config["encoder_units"], activation="relu"),
        Dropout(config["dropout"]),
        Dense(config["bottleneck_units"], activation="relu"),
        Dense(config["encoder_units"], activation="relu"),
        Dense(input_dim, activation="linear"),
    ])

    model.compile(
        optimizer=Adam(
            learning_rate=config["learning_rate"]
        ),
        loss="mse",
    )

    return model


def select_anomaly_threshold(y_true, errors):
    """Choose the threshold with the highest validation F1."""
    errors = np.asarray(errors).reshape(-1)

    candidates = np.unique(
        np.concatenate([
            np.array([errors.min() - 1e-9]),
            np.percentile(
                errors, np.linspace(1, 99.5, 200)
            ),
            np.array([errors.max() + 1e-9]),
        ])
    )

    best_threshold = float(candidates[0])
    best_f1 = -1.0

    for threshold in candidates:
        metrics = evaluate_anomaly(
            y_true, errors, float(threshold)
        )

        if metrics["f1"] > best_f1:
            best_f1 = float(metrics["f1"])
            best_threshold = float(threshold)

    return best_threshold, best_f1


def log_config_start(
    task, architecture, config, number,
    selection_metric, direction
):
    """Log configuration parameters and selection metadata."""
    mlflow.set_tag("task", task)
    mlflow.set_tag("architecture", architecture)
    mlflow.set_tag("selection_metric", selection_metric)
    mlflow.log_param("configuration_number", number)
    mlflow.log_param("selection_metric", selection_metric)
    mlflow.log_param("selection_direction", direction)

    log_task_params(task, mlflow_config_values(config))


# ============================================================
# 4. TASK 1: DEMAND FORECASTING - LSTM
# ============================================================

def train_demand():
    print("\n" + "=" * 60)
    print("TASK 1: DEMAND FORECASTING")
    print("Model: LSTM")
    print("Selection metric: Original-scale validation MAE")
    print("=" * 60)

    mlflow.set_experiment("Supply_Chain_Demand_Forecasting")

    # 1. Load processed data
    data = load_demand_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    scaler = joblib.load(
        DEMAND_DIR / "demand_scaler.pkl"
    )

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)

    # 2. Train and compare LSTM configurations
    best_model = None
    best_config = None
    best_val_mae = float("inf")

    for number, config in enumerate(DEMAND_CONFIGS, start=1):
        print(f"\nDemand configuration {number}: {config}")

        with mlflow.start_run(
            run_name=f"demand_config_{number}"
        ):
            log_config_start(
                "demand_forecasting",
                "LSTM",
                config,
                number,
                "validation_mae",
                "minimize",
            )

            model = make_lstm(
                config,
                X_train.shape[1:]
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                callbacks=make_callbacks(
                    patience=5,
                    monitor="val_mae",
                    mode="min",
                ),
                verbose=0,
            )

            # Predict validation values in scaled units
            val_pred_scaled = model.predict(
                X_val, verbose=0
            )

            # Scaled validation metrics
            val_scaled_metrics = evaluate_demand(
                y_val,
                val_pred_scaled,
            )

            # Correct R² calculation
            val_scaled_metrics["r2"] = float(
                r2_score(
                    np.asarray(y_val).reshape(-1),
                    np.asarray(val_pred_scaled).reshape(-1),
                )
            )

            # Convert validation values to original units
            val_true_original = scaler.inverse_transform(
                np.asarray(y_val).reshape(-1, 1)
            ).ravel()

            val_pred_original = scaler.inverse_transform(
                np.asarray(val_pred_scaled).reshape(-1, 1)
            ).ravel()

            val_original_metrics = evaluate_demand(
                val_true_original,
                val_pred_original,
            )

            val_original_metrics["r2"] = float(
                r2_score(
                    val_true_original,
                    val_pred_original,
                )
            )

            # Log metrics for both scales
            log_task_metrics(
                val_scaled_metrics,
                prefix="val_scaled_",
            )

            log_task_metrics(
                val_original_metrics,
                prefix="val_original_",
            )

            selection_mae = val_original_metrics["mae"]

            mlflow.log_metric(
                "selection_mae",
                selection_mae,
            )

            mlflow.log_metric(
                "epochs_trained",
                len(history.history["loss"]),
            )

            print(
                f"Scaled validation MSE: "
                f"{val_scaled_metrics['mse']:.6f}"
            )

            print(
                f"Original validation MAE: "
                f"{selection_mae:.4f}"
            )

            # Keep the model with the lowest validation MAE
            if selection_mae < best_val_mae:
                best_val_mae = selection_mae
                best_config = config.copy()
                best_model = model

    if best_model is None:
        raise RuntimeError(
            "No demand model was trained successfully."
        )

    # 3. Save the best model
    best_path = DEMAND_DIR / "best_demand_lstm.keras"
    save_model(best_model, best_path)

    # 4. Evaluate on the test set
    test_pred_scaled = best_model.predict(
        X_test, verbose=0
    )

    test_scaled_metrics = evaluate_demand(
        y_test,
        test_pred_scaled,
    )

    test_scaled_metrics["r2"] = float(
        r2_score(
            np.asarray(y_test).reshape(-1),
            np.asarray(test_pred_scaled).reshape(-1),
        )
    )

    # Convert predictions to original demand units
    test_true_original = scaler.inverse_transform(
        np.asarray(y_test).reshape(-1, 1)
    ).ravel()

    test_pred_original = scaler.inverse_transform(
        np.asarray(test_pred_scaled).reshape(-1, 1)
    ).ravel()

    test_original_metrics = evaluate_demand(
        test_true_original,
        test_pred_original,
    )

    test_original_metrics["r2"] = float(
        r2_score(
            test_true_original,
            test_pred_original,
        )
    )

    # 5. Baseline: previous observed demand
    # Assumes X_test shape is (samples, 7, 1)
    # and the only input feature is scaled demand.
    if X_test.ndim != 3 or X_test.shape[2] != 1:
        raise ValueError(
            "Persistence baseline expects X_test with "
            "shape (samples, timesteps, 1). "
            "Check your demand preprocessing."
        )

    baseline_pred_scaled = X_test[:, -1, 0].reshape(-1, 1)

    baseline_pred_original = scaler.inverse_transform(
        baseline_pred_scaled
    ).ravel()

    baseline_metrics = evaluate_demand(
        test_true_original,
        baseline_pred_original,
    )

    baseline_metrics["r2"] = float(
        r2_score(
            test_true_original,
            baseline_pred_original,
        )
    )

    print("\n" + "-" * 60)
    print("DEMAND MODEL COMPARISON")
    print("-" * 60)
    print("LSTM test metrics:", test_original_metrics)
    print("Baseline test metrics:", baseline_metrics)

    # 6. Log final test results to MLflow
    with mlflow.start_run(
        run_name="demand_best_test_evaluation"
    ):
        mlflow.set_tag("task", "demand_forecasting")
        mlflow.set_tag("selected_architecture", "LSTM")
        mlflow.set_tag("selection_metric", "validation_mae")

        log_task_params(
            "demand_forecasting",
            best_config,
        )

        mlflow.log_param("selection_metric", "validation_mae")
        mlflow.log_param("selection_direction", "minimize")

        mlflow.log_metric("best_val_mae", best_val_mae)

        log_task_metrics(
            test_scaled_metrics,
            prefix="test_scaled_",
        )

        log_task_metrics(
            test_original_metrics,
            prefix="test_original_",
        )

        log_task_metrics(
            test_original_metrics,
            prefix="test_",
        )

        log_task_metrics(
            baseline_metrics,
            prefix="baseline_test_",
        )

        mlflow.log_artifact(
            str(best_path),
            artifact_path="best_model",
        )

        mlflow.log_artifact(
            str(DEMAND_DIR / "demand_scaler.pkl"),
            artifact_path="preprocessing",
        )

    # 7. Print final results
    print("\nBest demand configuration:", best_config)
    print(f"Best validation MAE: {best_val_mae:.4f}")

    print(
        f"Test scaled MSE: "
        f"{test_scaled_metrics['mse']:.6f}"
    )

    print(
        f"Test original-scale MSE: "
        f"{test_original_metrics['mse']:.4f}"
    )

    print(
        f"Test original-scale RMSE: "
        f"{test_original_metrics['rmse']:.4f}"
    )

    print(
        f"Test original-scale MAE: "
        f"{test_original_metrics['mae']:.4f}"
    )

    print(
        f"Baseline test MAE: "
        f"{baseline_metrics['mae']:.4f}"
    )

    print("Saved model:", best_path)


# ============================================================
# 5. TASK 2: STOCKOUT PREDICTION - MLP
# ============================================================

def train_stockout():
    print("\n" + "=" * 60)
    print("TASK 2: STOCKOUT PREDICTION")
    print("Selection metric: validation recall (maximize)")
    print("=" * 60)

    mlflow.set_experiment("Supply_Chain_Stockout_Prediction")

    data = load_stockout_data()

    X_full = data["X_train"]
    y_full = np.asarray(data["y_train"]).reshape(-1).astype(int)

    X_test = data["X_test"]
    y_test = np.asarray(data["y_test"]).reshape(-1).astype(int)

    # Create validation data only from the training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=0.20,
        random_state=SEED,
        stratify=y_full,
    )

    class_weights = get_class_weights(y_train)

    best_model = None
    best_config = None
    best_val_recall = -1.0

    for number, config in enumerate(STOCKOUT_CONFIGS, start=1):
        print(f"\nStockout configuration {number}: {config}")

        with mlflow.start_run(
            run_name=f"stockout_config_{number}"
        ):
            log_config_start(
                "stockout_prediction",
                "MLP",
                config,
                number,
                "validation_recall",
                "maximize",
            )

            model = make_mlp(
                config,
                X_train.shape[1:]
            )

            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                class_weight=class_weights,
                callbacks=make_callbacks(
                    patience=5,
                    monitor="val_loss",
                    mode="min",
                ),
                verbose=0,
            )

            val_prob = model.predict(
                X_val, verbose=0
            ).reshape(-1)

            val_metrics = evaluate_stockout(
                y_val,
                val_prob,
                threshold=0.5,
            )

            log_task_metrics(val_metrics, prefix="val_")

            mlflow.log_metric(
                "selection_recall",
                val_metrics["recall"],
            )

            mlflow.log_metric(
                "epochs_trained",
                len(history.history["loss"]),
            )

            mlflow.log_param("classification_threshold", 0.5)

            print(
                f"Validation recall: {val_metrics['recall']:.4f}"
            )

            print(
                f"Validation precision: "
                f"{val_metrics['precision']:.4f}"
            )

            print(f"Validation F1: {val_metrics['f1']:.4f}")

            if val_metrics["recall"] > best_val_recall:
                best_val_recall = val_metrics["recall"]
                best_config = config.copy()
                best_model = model

    if best_model is None or best_config is None:
        raise RuntimeError("No stockout model was trained.")

    best_path = STOCKOUT_DIR / "best_stockout_mlp.keras"
    save_model(best_model, best_path)

    test_prob = best_model.predict(
        X_test, verbose=0
    ).reshape(-1)

    test_metrics = evaluate_stockout(
        y_test,
        test_prob,
        threshold=0.5,
    )

    with mlflow.start_run(
        run_name="stockout_best_test_evaluation"
    ):
        mlflow.set_tag("task", "stockout_prediction")
        mlflow.set_tag("selected_architecture", "MLP")
        mlflow.set_tag("selection_metric", "validation_recall")

        log_task_params(
            "stockout_prediction",
            mlflow_config_values(best_config),
        )

        mlflow.log_param("selection_metric", "validation_recall")
        mlflow.log_param("selection_direction", "maximize")
        mlflow.log_param("classification_threshold", 0.5)
        mlflow.log_metric("best_val_recall", best_val_recall)

        log_task_metrics(test_metrics, prefix="test_")
        mlflow.log_artifact(str(best_path), artifact_path="best_model")

    print("\nBest stockout configuration:", best_config)
    print(f"Best validation recall: {best_val_recall:.4f}")
    print("Stockout test metrics:", test_metrics)
    print("Saved model:", best_path)


# ============================================================
# 6. TASK 3: ANOMALY DETECTION - AUTOENCODER
# ============================================================

def train_anomaly():
    print("\n" + "=" * 60)
    print("TASK 3: ANOMALY DETECTION")
    print("Selection metric: validation F1-score (maximize)")
    print("=" * 60)

    mlflow.set_experiment("Supply_Chain_Anomaly_Detection")

    data = load_anomaly_data()

    # These keys must match preprocessing.py
    X_train_all = data["X_train_all"]
    y_train_all = np.asarray(data["y_train"]).reshape(-1).astype(int)

    X_test = data["X_test"]
    y_test = np.asarray(data["y_test"]).reshape(-1).astype(int)

    # Create validation data from the training split
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_all,
    )

    # Train the autoencoder only on normal training records
    X_fit_normal = X_fit[y_fit == 0]
    X_val_normal = X_val[y_val == 0]

    if len(X_fit_normal) == 0 or len(X_val_normal) == 0:
        raise ValueError(
            "Training and validation must contain normal records."
        )

    best_model = None
    best_config = None
    best_threshold = None
    best_val_f1 = -1.0

    for number, config in enumerate(ANOMALY_CONFIGS, start=1):
        print(f"\nAnomaly configuration {number}: {config}")

        with mlflow.start_run(
            run_name=f"anomaly_config_{number}"
        ):
            log_config_start(
                "anomaly_detection",
                "Autoencoder",
                config,
                number,
                "validation_f1",
                "maximize",
            )

            model = make_autoencoder(
                config,
                X_fit_normal.shape[1]
            )

            history = model.fit(
                X_fit_normal,
                X_fit_normal,
                validation_data=(
                    X_val_normal,
                    X_val_normal,
                ),
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                callbacks=make_callbacks(
                    patience=5,
                    monitor="val_loss",
                    mode="min",
                ),
                verbose=0,
            )

            # Calculate reconstruction errors for all validation records
            val_errors = calculate_reconstruction_errors(
                model,
                X_val,
            )

            threshold, _ = select_anomaly_threshold(
                y_val,
                val_errors,
            )

            val_metrics = evaluate_anomaly(
                y_val,
                val_errors,
                threshold,
            )

            log_task_metrics(val_metrics, prefix="val_")

            mlflow.log_metric(
                "selection_f1",
                val_metrics["f1"],
            )

            mlflow.log_metric(
                "epochs_trained",
                len(history.history["loss"]),
            )

            mlflow.log_metric(
                "anomaly_threshold",
                float(threshold),
            )

            print(f"Validation F1: {val_metrics['f1']:.4f}")

            print(
                f"Validation precision: "
                f"{val_metrics['precision']:.4f}"
            )

            print(
                f"Validation recall: "
                f"{val_metrics['recall']:.4f}"
            )

            print(f"Threshold: {threshold:.6f}")

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_config = config.copy()
                best_model = model
                best_threshold = float(threshold)

    if (
        best_model is None
        or best_config is None
        or best_threshold is None
    ):
        raise RuntimeError("No anomaly model was trained.")

    best_path = (
        ANOMALY_DIR / "best_anomaly_autoencoder.keras"
    )

    threshold_path = (
        ANOMALY_DIR / "best_anomaly_threshold.npy"
    )

    save_model(best_model, best_path)
    np.save(threshold_path, np.asarray(best_threshold))

    # Evaluate the test set without tuning the threshold on test data
    test_errors = calculate_reconstruction_errors(
        best_model,
        X_test,
    )

    test_metrics = evaluate_anomaly(
        y_test,
        test_errors,
        best_threshold,
    )

    with mlflow.start_run(
        run_name="anomaly_best_test_evaluation"
    ):
        mlflow.set_tag("task", "anomaly_detection")
        mlflow.set_tag("selected_architecture", "Autoencoder")
        mlflow.set_tag("selection_metric", "validation_f1")

        log_task_params(
            "anomaly_detection",
            best_config,
        )

        mlflow.log_param("selection_metric", "validation_f1")
        mlflow.log_param("selection_direction", "maximize")
        mlflow.log_metric("best_val_f1", best_val_f1)

        mlflow.log_metric(
            "anomaly_threshold",
            best_threshold,
        )

        log_task_metrics(test_metrics, prefix="test_")

        mlflow.log_artifact(
            str(best_path),
            artifact_path="best_model",
        )

        mlflow.log_artifact(
            str(threshold_path),
            artifact_path="best_model",
        )

    print("\nBest anomaly configuration:", best_config)
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best threshold: {best_threshold:.6f}")
    print("Anomaly test metrics:", test_metrics)
    print("Saved model:", best_path)
    print("Saved threshold:", threshold_path)


# ============================================================
# 7. RUN ALL THREE TASKS
# ============================================================

if __name__ == "__main__":
    train_demand()
    train_stockout()
    train_anomaly()

    print("\n" + "=" * 60)
    print("ALL THREE TRAINING TASKS COMPLETED")
    print("=" * 60)



