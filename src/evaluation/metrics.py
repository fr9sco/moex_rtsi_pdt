from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(y_true, y_pred, prefix: str | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate on an empty target")

    accuracy = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    specificity = _safe_div(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision_up": precision,
        "recall_up": recall,
        "f1_up": f1,
        "true_positive_share": float(y_true.mean()),
        "predicted_positive_share": float(y_pred.mean()),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }
    if prefix is None:
        return metrics
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def majority_baseline(train_y, n_predictions: int) -> np.ndarray:
    values = pd.Series(train_y).astype(int)
    majority_class = int(values.mode().iloc[0])
    return np.full(n_predictions, majority_class, dtype=int)


def last_direction_baseline(frame: pd.DataFrame) -> np.ndarray:
    return (frame["return_1d"].to_numpy() > 0).astype(int)


def regression_metrics(y_true, y_pred, prefix: str | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate on an empty target")

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    non_zero_mask = y_true != 0
    mape_pct = float(np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape_pct,
    }
    if prefix is None:
        return metrics
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def combined_forecast_metrics(
    frame: pd.DataFrame,
    predicted_close,
    predicted_return,
    prefix: str | None = None,
) -> dict[str, float]:
    predicted_close = np.asarray(predicted_close, dtype=float)
    predicted_return = np.asarray(predicted_return, dtype=float)
    predicted_up = (predicted_return > 0).astype(int)

    metrics = {}
    metrics.update(regression_metrics(frame["future_close"], predicted_close))
    metrics.update(classification_metrics(frame["target_up"], predicted_up))
    if prefix is None:
        return metrics
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)
