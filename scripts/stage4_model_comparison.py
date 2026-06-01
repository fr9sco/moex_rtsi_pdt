from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import combined_forecast_metrics
from src.features.technical import get_default_feature_columns
from src.models.pdt import PermutationDecisionTreeClassifier
from src.pipelines.sequences import build_lstm_sequences
from src.pipelines.supervised import split_dataset


RANDOM_STATE = 42
LOOKBACK = 30

XGB_PARAM_GRID = [
    {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 180,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
    },
    {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
    },
]

PDT_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "range_pct",
    "gap_pct",
    "intraday_return",
    "ma_5_gap",
    "ma_20_gap",
    "ma_50_gap",
    "rolling_vol_20d",
    "rsi_14",
    "position_20d",
    "value_change_1d",
]


class LSTMReturnRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="compare PDT, XGBoost and LSTM")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--lstm-max-epochs", type=int, default=35)
    parser.add_argument("--lstm-patience", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible_seed(RANDOM_STATE)

    tables_dir = ROOT / "reports" / "tables"
    figures_dir = ROOT / "reports" / "figures"
    for directory in [tables_dir, figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    feature_columns = get_default_feature_columns()
    model_rows = []
    prediction_frames = []
    xgb_tuning_rows = []
    lstm_training_rows = []
    xgb_importance_rows = []

    pdt_params = load_stage3_pdt_params(tables_dir)

    for horizon in args.horizons:
        dataset_path = ROOT / "data" / "processed" / f"moex_supervised_h{horizon}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Run scripts/stage3_build_datasets.py first: missing {dataset_path}"
            )
        dataset = pd.read_csv(dataset_path, parse_dates=["date", "future_date"])

        for secid, security_dataset in dataset.groupby("secid", sort=True):
            print(f"Stage 4 models: secid={secid}, horizon={horizon}", flush=True)
            security_dataset = security_dataset.sort_values("date").reset_index(drop=True)
            splits = split_dataset(security_dataset)

            baseline_predictions = predict_baselines(splits.test, horizon)
            for model_name, pred_return in baseline_predictions.items():
                pred_close = splits.test["close"].to_numpy() * (1 + pred_return)
                append_metrics_and_predictions(
                    rows=model_rows,
                    prediction_frames=prediction_frames,
                    horizon=horizon,
                    secid=secid,
                    model=model_name,
                    frame=splits.test,
                    predicted_return=pred_return,
                    predicted_close=pred_close,
                )

            pdt_pred_return, pdt_test_frame = predict_pdt_returns(
                splits.train_validation,
                splits.test,
                horizon,
                secid,
                pdt_params,
            )
            append_metrics_and_predictions(
                rows=model_rows,
                prediction_frames=prediction_frames,
                horizon=horizon,
                secid=secid,
                model="PDT_direction_median_return",
                frame=pdt_test_frame,
                predicted_return=pdt_pred_return,
                predicted_close=pdt_test_frame["close"].to_numpy() * (1 + pdt_pred_return),
            )

            xgb_pred_return, xgb_tuning, xgb_model = fit_predict_xgboost(
                splits.train,
                splits.validation,
                splits.train_validation,
                splits.test,
                feature_columns,
                horizon,
                secid,
            )
            xgb_tuning_rows.extend(xgb_tuning)
            append_feature_importance(
                xgb_importance_rows,
                xgb_model,
                feature_columns,
                horizon,
                secid,
            )
            append_metrics_and_predictions(
                rows=model_rows,
                prediction_frames=prediction_frames,
                horizon=horizon,
                secid=secid,
                model="XGBoostRegressor",
                frame=splits.test,
                predicted_return=xgb_pred_return,
                predicted_close=splits.test["close"].to_numpy() * (1 + xgb_pred_return),
            )

            lstm_pred_return, lstm_test_frame, lstm_info = fit_predict_lstm(
                security_dataset,
                splits,
                feature_columns,
                horizon,
                secid,
                max_epochs=args.lstm_max_epochs,
                patience=args.lstm_patience,
            )
            lstm_training_rows.append(lstm_info)
            append_metrics_and_predictions(
                rows=model_rows,
                prediction_frames=prediction_frames,
                horizon=horizon,
                secid=secid,
                model="LSTMRegressor",
                frame=lstm_test_frame,
                predicted_return=lstm_pred_return,
                predicted_close=lstm_test_frame["close"].to_numpy() * (1 + lstm_pred_return),
            )

    comparison = pd.DataFrame(model_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    xgb_tuning = pd.DataFrame(xgb_tuning_rows)
    lstm_training = pd.DataFrame(lstm_training_rows)
    xgb_importance = pd.DataFrame(xgb_importance_rows)

    comparison.to_csv(tables_dir / "stage4_model_comparison.csv", index=False)
    predictions.to_csv(tables_dir / "stage4_predictions.csv", index=False)
    xgb_tuning.to_csv(tables_dir / "stage4_xgboost_validation.csv", index=False)
    lstm_training.to_csv(tables_dir / "stage4_lstm_training.csv", index=False)
    xgb_importance.to_csv(tables_dir / "stage4_xgboost_feature_importance.csv", index=False)

    create_figures(comparison, predictions, figures_dir)
    print(comparison.to_string(index=False))
    print(f"Saved Stage 4 tables -> {tables_dir}")
    print(f"Saved Stage 4 figures -> {figures_dir}")


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def predict_baselines(test: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    predictions = {
        "NaivePrice": np.zeros(len(test), dtype=float),
        "LastDailyReturn": test["return_1d"].to_numpy(dtype=float),
    }
    if horizon != 1:
        predictions["LastHorizonReturn"] = test[f"return_{horizon}d"].to_numpy(dtype=float)
    return predictions


def load_stage3_pdt_params(tables_dir: Path) -> pd.DataFrame:
    path = tables_dir / "stage3_pdt_best_params.csv"
    if not path.exists():
        raise FileNotFoundError("Stage 3 PDT params are required before Stage 4")
    return pd.read_csv(path)


def predict_pdt_returns(
    train_validation: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int,
    secid: str,
    pdt_params: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    params_row = pdt_params[
        (pdt_params["horizon"] == horizon) & (pdt_params["secid"] == secid)
    ].iloc[0]
    feature_columns = [feature for feature in PDT_FEATURE_COLUMNS if feature in train_validation.columns]

    model = PermutationDecisionTreeClassifier(
        max_depth=int(params_row["max_depth"]),
        min_samples_leaf=int(params_row["min_samples_leaf"]),
        min_samples_split=int(params_row["min_samples_leaf"]) * 2,
        max_thresholds=int(params_row["max_thresholds"]),
        normalize_etc=parse_bool(params_row.get("normalize_etc", False)),
    )
    model.fit(train_validation[feature_columns], train_validation["target_up"])
    predicted_up = model.predict(test[feature_columns]).astype(int)

    median_by_class = train_validation.groupby("target_up")["future_return"].median().to_dict()
    global_median = float(train_validation["future_return"].median())
    predicted_return = np.array(
        [median_by_class.get(label, global_median) for label in predicted_up],
        dtype=float,
    )
    return predicted_return, test


def parse_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def fit_predict_xgboost(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    train_validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    secid: str,
) -> tuple[np.ndarray, list[dict[str, float | int | str]], XGBRegressor]:
    tuning_rows = []
    best_score = math.inf
    best_params = None

    for param_id, params in enumerate(XGB_PARAM_GRID):
        model = make_xgb_model(params)
        model.fit(train[feature_columns], train["future_return"])
        val_pred_return = model.predict(validation[feature_columns])
        val_pred_close = validation["close"].to_numpy() * (1 + val_pred_return)
        metrics = combined_forecast_metrics(validation, val_pred_close, val_pred_return)
        row = {
            "horizon": horizon,
            "secid": secid,
            "param_id": param_id,
            **params,
            **metrics,
        }
        tuning_rows.append(row)
        if metrics["rmse"] < best_score:
            best_score = metrics["rmse"]
            best_params = params

    final_model = make_xgb_model(best_params)
    final_model.fit(train_validation[feature_columns], train_validation["future_return"])
    test_pred_return = final_model.predict(test[feature_columns])
    return test_pred_return, tuning_rows, final_model


def make_xgb_model(params: dict[str, float | int]) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=2,
        **params,
    )


def append_feature_importance(
    rows: list[dict[str, float | int | str]],
    model: XGBRegressor,
    feature_columns: list[str],
    horizon: int,
    secid: str,
) -> None:
    importances = model.feature_importances_
    for feature, importance in zip(feature_columns, importances):
        rows.append(
            {
                "horizon": horizon,
                "secid": secid,
                "feature": feature,
                "importance": float(importance),
            }
        )


def fit_predict_lstm(
    security_dataset: pd.DataFrame,
    splits,
    feature_columns: list[str],
    horizon: int,
    secid: str,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float | int | str]]:
    full = security_dataset.sort_values("date").reset_index(drop=True).copy()

    train_scaled, train_mean, train_std = scale_for_lstm(full, splits.train, feature_columns)
    train_positions = positions_for_dates(full, splits.train)
    validation_positions = positions_for_dates(full, splits.validation)
    X_train, y_train, _ = build_lstm_sequences(
        train_scaled,
        feature_columns,
        "future_return",
        LOOKBACK,
        eligible_positions=train_positions,
    )
    X_val, y_val, _ = build_lstm_sequences(
        train_scaled,
        feature_columns,
        "future_return",
        LOOKBACK,
        eligible_positions=validation_positions,
    )

    y_train_scaled = ((y_train - train_mean) / train_std).astype(np.float32)
    y_val_scaled = ((y_val - train_mean) / train_std).astype(np.float32)

    model, best_epoch, best_val_loss = train_lstm_with_validation(
        X_train,
        y_train_scaled,
        X_val,
        y_val_scaled,
        max_epochs=max_epochs,
        patience=patience,
    )

    final_scaled, final_mean, final_std = scale_for_lstm(full, splits.train_validation, feature_columns)
    train_validation_positions = positions_for_dates(full, splits.train_validation)
    test_positions = positions_for_dates(full, splits.test)
    X_train_validation, y_train_validation, _ = build_lstm_sequences(
        final_scaled,
        feature_columns,
        "future_return",
        LOOKBACK,
        eligible_positions=train_validation_positions,
    )
    X_test, _, test_sequence_positions = build_lstm_sequences(
        final_scaled,
        feature_columns,
        "future_return",
        LOOKBACK,
        eligible_positions=test_positions,
    )

    y_train_validation_scaled = (
        (y_train_validation - final_mean) / final_std
    ).astype(np.float32)
    final_model = train_lstm_fixed_epochs(
        X_train_validation,
        y_train_validation_scaled,
        epochs=max(1, best_epoch),
    )
    pred_scaled = predict_lstm(final_model, X_test)
    pred_return = pred_scaled * final_std + final_mean
    test_frame = full.iloc[test_sequence_positions].reset_index(drop=True)

    info = {
        "horizon": horizon,
        "secid": secid,
        "lookback": LOOKBACK,
        "train_sequences": len(X_train),
        "validation_sequences": len(X_val),
        "train_validation_sequences": len(X_train_validation),
        "test_sequences": len(X_test),
        "best_epoch": best_epoch,
        "best_validation_loss_scaled": best_val_loss,
    }
    return pred_return, test_frame, info


def scale_for_lstm(
    full: pd.DataFrame,
    scaler_fit_frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, float, float]:
    scaler = StandardScaler()
    scaler.fit(scaler_fit_frame[feature_columns])

    scaled = full.copy()
    scaled[feature_columns] = scaler.transform(full[feature_columns])
    target_mean = float(scaler_fit_frame["future_return"].mean())
    target_std = float(scaler_fit_frame["future_return"].std())
    if target_std == 0 or not np.isfinite(target_std):
        target_std = 1.0
    return scaled, target_mean, target_std


def positions_for_dates(full: pd.DataFrame, split_frame: pd.DataFrame) -> set[int]:
    dates = set(pd.to_datetime(split_frame["date"]))
    return set(full.index[full["date"].isin(dates)].tolist())


def train_lstm_with_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int,
    patience: int,
) -> tuple[LSTMReturnRegressor, int, float]:
    model = LSTMReturnRegressor(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, y_train, shuffle=True)

    best_state = None
    best_epoch = 1
    best_val_loss = math.inf
    rounds_without_improvement = 0

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_tensor), y_val_tensor).item())

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_val_loss


def train_lstm_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
) -> LSTMReturnRegressor:
    model = LSTMReturnRegressor(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(X_train, y_train, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    return model


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=64, shuffle=shuffle, generator=generator)


def predict_lstm(model: LSTMReturnRegressor, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(X, dtype=torch.float32)).numpy()
    return prediction.astype(float)


def append_metrics_and_predictions(
    rows: list[dict[str, float | int | str]],
    prediction_frames: list[pd.DataFrame],
    horizon: int,
    secid: str,
    model: str,
    frame: pd.DataFrame,
    predicted_return: np.ndarray,
    predicted_close: np.ndarray,
) -> None:
    metrics = combined_forecast_metrics(frame, predicted_close, predicted_return)
    rows.append(
        {
            "horizon": horizon,
            "secid": secid,
            "model": model,
            "rows": len(frame),
            "test_first_date": frame["date"].min().date().isoformat(),
            "test_last_date": frame["date"].max().date().isoformat(),
            **metrics,
        }
    )

    output = frame[
        [
            "secid",
            "date",
            "future_date",
            "close",
            "future_close",
            "future_return",
            "target_up",
        ]
    ].copy()
    output["horizon"] = horizon
    output["model"] = model
    output["predicted_return"] = predicted_return
    output["predicted_close"] = predicted_close
    output["predicted_up"] = (predicted_return > 0).astype(int)
    prediction_frames.append(output)


def create_figures(comparison: pd.DataFrame, predictions: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plot_data = comparison.copy()
    plot_data["task"] = plot_data["secid"] + ", h=" + plot_data["horizon"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_data, x="task", y="accuracy", hue="model", ax=ax)
    ax.set_title("Stage 4 test directional accuracy")
    ax.set_ylim(0.35, 0.65)
    ax.set_xlabel("")
    ax.set_ylabel("Directional accuracy")
    ax.legend(title="", ncols=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "stage4_directional_accuracy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_data, x="task", y="rmse", hue="model", ax=ax)
    ax.set_title("Stage 4 test RMSE by model")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE, index points")
    ax.legend(title="", ncols=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "stage4_rmse.png", dpi=180)
    plt.close(fig)

    subset = predictions[
        (predictions["secid"] == "IMOEX")
        & (predictions["horizon"] == 1)
        & (predictions["model"].isin(["XGBoostRegressor", "LSTMRegressor", "NaivePrice"]))
    ].copy()
    if not subset.empty:
        fig, ax = plt.subplots(figsize=(13, 6))
        true_frame = subset.drop_duplicates("date").sort_values("date")
        ax.plot(true_frame["date"], true_frame["future_close"], label="True future close", linewidth=1.4)
        for model_name, group in subset.groupby("model", sort=True):
            group = group.sort_values("date")
            ax.plot(group["date"], group["predicted_close"], label=model_name, linewidth=1.0)
        ax.set_title("IMOEX h=1: true and predicted future close")
        ax.set_ylabel("Index points")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "stage4_predictions_imoex_h1.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
