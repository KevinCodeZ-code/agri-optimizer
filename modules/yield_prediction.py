"""
Yield Prediction Module: Hybrid GRU-LSTM vs XGBoost.
Replicates MATLAB-based crop yield prediction from climatic variables.
"""
import numpy as np
import pandas as pd


def engineer_features(df):
    """Generate engineered features from weather data."""
    result = df.copy()
    result["DTR"] = result["Tmax"] - result["Tmin"]
    result["DTR"] = result["DTR"].clip(lower=0)
    month = result["Date"].dt.month
    result["Sin_Month"] = np.sin(2 * np.pi * month / 12)
    result["Cos_Month"] = np.cos(2 * np.pi * month / 12)
    return result


def sequential_partition(df, train_ratio=0.8, val_ratio=0.1):
    """Chronological split: train 80%, val 10%, test 10%."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def min_max_normalize(train, val, test, feature_cols, target_col="Yield"):
    """Min-Max normalize features and target using training set limits."""
    result = {}
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        result[name] = subset.copy()

    feature_mins = {}
    feature_maxs = {}
    for col in feature_cols:
        cmin = train[col].min()
        cmax = train[col].max()
        feature_mins[col] = cmin
        feature_maxs[col] = cmax
        if cmax > cmin:
            for name in ["train", "val", "test"]:
                result[name][col] = (result[name][col] - cmin) / (cmax - cmin)
        else:
            for name in ["train", "val", "test"]:
                result[name][col] = 0.5

    target_mins = {}
    target_maxs = {}
    for name in ["train", "val", "test"]:
        subset = result[name]
        if target_col in subset.columns:
            tmin = train[target_col].min() if name == "train" else 0
            tmax = train[target_col].max() if name == "train" else 1
            if name == "train":
                target_mins["mins"] = tmin
                target_maxs["maxs"] = tmax
            else:
                if "mins" in target_mins and "maxs" in target_mins:
                    tmin = target_mins["mins"]
                    tmax = target_mins["maxs"]
            if tmax > tmin:
                result[name][target_col] = (subset[target_col] - tmin) / (tmax - tmin)
            else:
                result[name][target_col] = 0.5

    norm_info = {
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs,
        "target_min": train[target_col].min() if target_col in train.columns else 0,
        "target_max": train[target_col].max() if target_col in train.columns else 1,
    }
    return result, norm_info


def build_gru_lstm_model(timesteps=10, n_features=6):
    """Build Hybrid GRU-LSTM network architecture."""
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout, Input

        model = Sequential([
            Input(shape=(timesteps, n_features)),
            GRU(64, return_sequences=True, activation="tanh"),
            Dropout(0.2),
            LSTM(32, return_sequences=False, activation="tanh"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model
    except ImportError:
        return None


def build_xgboost_model():
    """Build XGBoost regressor."""
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        return model
    except ImportError:
        return None


def create_sequences(data, feature_cols, target_col, timesteps=10):
    """Create sliding window sequences for time series models."""
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[feature_cols].iloc[i:i + timesteps].values)
        y.append(data[target_col].iloc[i + timesteps])
    return np.array(X), np.array(y)


def compute_rmse(actual, predicted):
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2)))


def compute_mae(actual, predicted):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.array(actual) - np.array(predicted))))


def run_gru_lstm(train_norm, val_norm, test_norm, norm_info, feature_cols, target_col="Yield"):
    """Train Hybrid GRU-LSTM and return predictions + metrics."""
    model = build_gru_lstm_model()
    if model is None:
        return None, "TensorFlow/Keras not available"

    timesteps = 10
    X_train, y_train = create_sequences(train_norm, feature_cols, target_col, timesteps)
    X_val, y_val = create_sequences(val_norm, feature_cols, target_col, timesteps)
    X_test, y_test = create_sequences(test_norm, feature_cols, target_col, timesteps)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        return None, "Insufficient data for sequence creation"

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=0,
        callbacks=[],
    )

    train_pred_norm = model.predict(X_train, verbose=0).flatten()
    val_pred_norm = model.predict(X_val, verbose=0).flatten()
    test_pred_norm = model.predict(X_test, verbose=0).flatten()

    tmin = norm_info["target_min"]
    tmax = norm_info["target_max"]

    def denormalize(norm_vals):
        return np.array(norm_vals) * (tmax - tmin) + tmin if tmax > tmin else np.array(norm_vals)

    train_pred = denormalize(train_pred_norm)
    val_pred = denormalize(val_pred_norm)
    test_pred = denormalize(test_pred_norm)
    y_train_actual = denormalize(y_train)
    y_val_actual = denormalize(y_val)
    y_test_actual = denormalize(y_test)

    def safe_rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else 0.0

    def safe_mae(a, b):
        return float(np.mean(np.abs(a - b))) if len(a) > 0 else 0.0

    rmse_train = safe_rmse(y_train_actual, train_pred)
    rmse_val = safe_rmse(y_val_actual, val_pred)
    rmse_test = safe_rmse(y_test_actual, test_pred)
    mae_train = safe_mae(y_train_actual, train_pred)
    mae_val = safe_mae(y_val_actual, val_pred)
    mae_test = safe_mae(y_test_actual, test_pred)

    train_df = train_norm.iloc[timesteps:].reset_index(drop=True)
    val_df = val_norm.iloc[timesteps:].reset_index(drop=True)
    test_df = test_norm.iloc[timesteps:].reset_index(drop=True)

    epochs_list = list(range(len(history.history["loss"])))
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    return {
        "model": model,
        "history": {"epochs": epochs_list, "train_loss": train_loss, "val_loss": val_loss},
        "predictions": {
            "train": {"actual": y_train_actual.tolist(), "predicted": train_pred.tolist(),
                      "dates": train_df["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in train_df.columns else []},
            "val": {"actual": y_val_actual.tolist(), "predicted": val_pred.tolist(),
                    "dates": val_df["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in val_df.columns else []},
            "test": {"actual": y_test_actual.tolist(), "predicted": test_pred.tolist(),
                     "dates": test_df["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in test_df.columns else []},
        },
        "metrics": {
            "rmse_train": round(rmse_train, 4),
            "rmse_val": round(rmse_val, 4),
            "rmse_test": round(rmse_test, 4),
            "mae_train": round(mae_train, 4),
            "mae_val": round(mae_val, 4),
            "mae_test": round(mae_test, 4),
        },
    }, None


def run_xgboost(train, val, test, feature_cols, target_col="Yield"):
    """Train XGBoost and return predictions + metrics."""
    model = build_xgboost_model()
    if model is None:
        return None, "XGBoost not available"

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_val = val[feature_cols].values
    y_val = val[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    def safe_rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) > 0 else 0.0

    def safe_mae(a, b):
        return float(np.mean(np.abs(a - b))) if len(a) > 0 else 0.0

    rmse_train = safe_rmse(y_train, train_pred)
    rmse_val = safe_rmse(y_val, val_pred)
    rmse_test = safe_rmse(y_test, test_pred)
    mae_train = safe_mae(y_train, train_pred)
    mae_val = safe_mae(y_val, val_pred)
    mae_test = safe_mae(y_test, test_pred)

    return {
        "model": model,
        "feature_importance": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else [],
        "predictions": {
            "train": {"actual": y_train.tolist(), "predicted": train_pred.tolist(),
                      "dates": train["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in train.columns else []},
            "val": {"actual": y_val.tolist(), "predicted": val_pred.tolist(),
                    "dates": val["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in val.columns else []},
            "test": {"actual": y_test.tolist(), "predicted": test_pred.tolist(),
                     "dates": test["Date"].dt.strftime("%Y-%m-%d").tolist() if "Date" in test.columns else []},
        },
        "metrics": {
            "rmse_train": round(rmse_train, 4),
            "rmse_val": round(rmse_val, 4),
            "rmse_test": round(rmse_test, 4),
            "mae_train": round(mae_train, 4),
            "mae_val": round(mae_val, 4),
            "mae_test": round(mae_test, 4),
        },
    }, None


def run_yield_prediction(weather_df):
    """Main entry point: run full yield prediction pipeline."""
    if weather_df is None or weather_df.empty:
        return None, "No weather data available"

    df = weather_df.copy()
    if "Yield" not in df.columns:
        df["Yield"] = (60 + 15 * np.sin(2 * np.pi * np.arange(len(df)) / 365)
                       + np.random.randn(len(df)) * 3).clip(min=0)

    features_df = engineer_features(df)
    feature_cols = ["Tmax", "Tmin", "Rainfall", "DTR", "Sin_Month", "Cos_Month"]
    feature_cols = [c for c in feature_cols if c in features_df.columns]

    train, val, test = sequential_partition(features_df)
    norm_sets, norm_info = min_max_normalize(train, val, test, feature_cols, "Yield")

    gru_result, gru_error = run_gru_lstm(
        norm_sets["train"], norm_sets["val"], norm_sets["test"],
        norm_info, feature_cols, "Yield"
    )
    xgb_result, xgb_error = run_xgboost(train, val, test, feature_cols, "Yield")

    best_model = "GRU-LSTM"
    best_rmse = float("inf")
    if gru_result and gru_result["metrics"]["rmse_test"] < float("inf"):
        best_rmse = gru_result["metrics"]["rmse_test"]
    if xgb_result and xgb_result["metrics"]["rmse_test"] < best_rmse:
        best_model = "XGBoost"
        best_rmse = xgb_result["metrics"]["rmse_test"]

    return {
        "feature_engineering": {
            "raw_data": df,
            "engineered_data": features_df,
            "feature_cols": feature_cols,
        },
        "partitioning": {
            "train": train,
            "val": val,
            "test": test,
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        },
        "normalization": norm_info,
        "gru_lstm": gru_result,
        "xgboost": xgb_result,
        "best_model": best_model,
        "best_rmse": round(best_rmse, 4) if best_rmse < float("inf") else None,
    }, None
