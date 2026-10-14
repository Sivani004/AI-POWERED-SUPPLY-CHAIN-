
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
)


# =========================================================
# PROJECT SETTINGS
# =========================================================


ROOT = Path(__file__).resolve().parent.parent


SUPPLY_FILE = ROOT / "data" / "supply_chain.csv"
ANOMALY_FILE = ROOT / "data" / "supply_chain_anomaly_6000.csv"


OUTPUT_DIR = ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
WINDOW_SIZE = 7


def load_data(path):
    """Load CSV and standardize column names."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    print(f"\nLoaded: {path.name}")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    return df


def encode_binary(series, column_name):
    """Convert common Yes/No and 0/1 labels into integers."""
    normalized = series.astype("string").str.strip().str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    result = normalized.map(mapping)

    if result.isna().any():
        invalid = normalized[result.isna()].unique().tolist()
        raise ValueError(
            f"Unexpected labels in '{column_name}': {invalid}"
        )

    return result.astype(int)


def save_arrays(folder, filename, **arrays):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / filename, **arrays)
    print("Saved:", folder / filename)


# =========================================================
# TASK 1: LSTM - DEMAND FORECASTING
# =========================================================

def preprocess_demand():
    print("\n" + "=" * 55)
    print("TASK 1: LSTM DEMAND FORECASTING")
    print("=" * 55)

    df = load_data(SUPPLY_FILE)

    required = ["date", "demand"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Demand dataset needs columns {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert columns to appropriate data types.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["demand"] = pd.to_numeric(df["demand"], errors="coerce")

    # Remove rows with invalid dates or demand.
    df = df.dropna(subset=["date", "demand"]).copy()
    df = df.sort_values("date")

    # Create one daily demand value across all products.
    daily = (
        df.groupby("date", as_index=False)["demand"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(daily) <= WINDOW_SIZE + 2:
        raise ValueError(
            "Not enough daily observations for forecasting."
        )

    # Chronological split: 70% training, 15% validation, 15% test.
    n = len(daily)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    if train_end <= WINDOW_SIZE:
        raise ValueError(
            "Not enough training observations for the selected window."
        )

    # Fit scaler using training data only.
    scaler = MinMaxScaler()
    scaler.fit(daily[["demand"]].iloc[:train_end])

    scaled = scaler.transform(
        daily[["demand"]]
    ).astype(np.float32).ravel()

    # Prepare sequences: previous 7 days -> next day.
    X, y = [], []

    for i in range(WINDOW_SIZE, len(scaled)):
        X.append(scaled[i - WINDOW_SIZE:i])
        y.append(scaled[i])

    X = np.asarray(X, dtype=np.float32).reshape(
        -1, WINDOW_SIZE, 1
    )
    y = np.asarray(y, dtype=np.float32)

    target_indices = np.arange(WINDOW_SIZE, len(daily))

    train_mask = target_indices < train_end
    val_mask = (
        (target_indices >= train_end)
        & (target_indices < val_end)
    )
    test_mask = target_indices >= val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    folder = OUTPUT_DIR / "demand"
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, folder / "demand_scaler.pkl")

    save_arrays(
        folder,
        "demand_processed.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    print("Daily records:", len(daily))
    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Testing shape:", X_test.shape)


# =========================================================
# TASK 2: MLP - STOCKOUT PREDICTION
# =========================================================

def preprocess_stockout():
    print("\n" + "=" * 55)
    print("TASK 2: MLP STOCKOUT PREDICTION")
    print("=" * 55)

    df = load_data(SUPPLY_FILE)

    target = "stockout"

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[target]).copy()

    y = encode_binary(df[target], target)

    if y.nunique() != 2:
        raise ValueError(
            "Stockout target must contain both 0 and 1."
        )

    # Exclude targets, labels, identifiers, and date fields.
    excluded = {
        "stockout",
        "is_anomaly",
        "anomaly_type",
        "date",
        "timestamp",
        "order_date",
        "record_id",
        "product_id",
        "supplier_id",
        "customer_id",
        "warehouse_id",
        "order_id",
        "id",
    }

    X = df.drop(
        columns=[c for c in df.columns if c in excluded]
    ).copy()

    if X.empty:
        raise ValueError("No feature columns remain for stockout.")

    # Convert numeric-looking text columns into numeric columns.
    for col in X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist():

        converted = pd.to_numeric(X[col], errors="coerce")

        if converted.notna().mean() > 0.90:
            X[col] = converted

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]

    # Stratified split preserves the proportion of stockout labels.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    transformers = []

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("numeric", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ])

        transformers.append(
            ("categorical", categorical_pipeline, categorical_cols)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # Fit transformations on training data only.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Dense arrays are convenient for a basic TensorFlow MLP.
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()

    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    X_train_processed = X_train_processed.astype(np.float32)
    X_test_processed = X_test_processed.astype(np.float32)

    y_train = y_train.to_numpy(dtype=np.float32)
    y_test = y_test.to_numpy(dtype=np.float32)

    folder = OUTPUT_DIR / "stockout"
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        preprocessor,
        folder / "stockout_preprocessor.pkl",
    )

    save_arrays(
        folder,
        "stockout_processed.npz",
        X_train=X_train_processed,
        y_train=y_train,
        X_test=X_test_processed,
        y_test=y_test,
    )

    print("Training features:", X_train_processed.shape)
    print("Testing features:", X_test_processed.shape)
    print("Training stockout rate:", round(float(y_train.mean()), 4))


# =========================================================
# TASK 3: AUTOENCODER - ANOMALY DETECTION
# =========================================================

def preprocess_anomaly():
    print("\n" + "=" * 55)
    print("TASK 3: AUTOENCODER ANOMALY DETECTION")
    print("=" * 55)

    df = load_data(ANOMALY_FILE)

    features = [
        "demand",
        "inventory_level",
        "orders",
        "price",
        "lead_time_days",
        "supplier_rating",
        "transport_cost",
        "warehouse_capacity",
    ]

    target = "is_anomaly"

    required = features + [target]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Anomaly dataset is missing {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Convert model features into numeric values.
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=features + [target]).copy()

    y = encode_binary(df[target], target)

    if y.nunique() != 2:
        raise ValueError(
            "Anomaly labels must contain both normal and anomaly records."
        )

    # Do not remove outliers; they are the records we want to detect.
    X = df[features].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # Train the scaler on normal training records only.
    normal_mask = y_train == 0
    X_train_normal_raw = X_train.iloc[
        np.flatnonzero(normal_mask)
    ]

    if len(X_train_normal_raw) < 2:
        raise ValueError(
            "Not enough normal training samples for scaling."
        )

    scaler = StandardScaler()
    scaler.fit(X_train_normal_raw)

    X_train_scaled = scaler.transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Train the Autoencoder on normal records only.
    X_train_normal = X_train_scaled[normal_mask]

    folder = OUTPUT_DIR / "anomaly"
    folder.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, folder / "anomaly_scaler.pkl")

    save_arrays(
        folder,
        "anomaly_processed.npz",
        X_train_normal=X_train_normal,
        X_train_all=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
    )

    print("Normal training samples:", X_train_normal.shape)
    print("All training samples:", X_train_scaled.shape)
    print("Testing samples:", X_test_scaled.shape)


# =========================================================
# RUN ALL THREE TASKS
# =========================================================

if __name__ == "__main__":
    print("\nSUPPLY CHAIN PROJECT PREPROCESSING")
    print("=" * 55)

    preprocess_demand()
    preprocess_stockout()
    preprocess_anomaly()

    print("\nAll three preprocessing tasks completed!")
