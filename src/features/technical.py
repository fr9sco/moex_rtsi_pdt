from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "log_return_1d",
    "range_pct",
    "gap_pct",
    "intraday_return",
    "body_pct",
    "high_close_gap",
    "close_low_gap",
    "ma_5_gap",
    "ma_10_gap",
    "ma_20_gap",
    "ma_50_gap",
    "rolling_vol_5d",
    "rolling_vol_10d",
    "rolling_vol_20d",
    "rsi_14",
    "position_20d",
    "drawdown_20d",
    "value_change_1d",
]


def get_default_feature_columns() -> list[str]:
    return list(DEFAULT_FEATURE_COLUMNS)


def add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("date").copy()

    data["return_1d"] = data["close"].pct_change(1)
    data["return_2d"] = data["close"].pct_change(2)
    data["return_5d"] = data["close"].pct_change(5)
    data["return_10d"] = data["close"].pct_change(10)
    data["return_20d"] = data["close"].pct_change(20)
    data["log_return_1d"] = np.log(data["close"] / data["close"].shift(1))

    data["range_pct"] = (data["high"] - data["low"]) / data["close"]
    data["gap_pct"] = data["open"] / data["close"].shift(1) - 1
    data["intraday_return"] = data["close"] / data["open"] - 1
    data["body_pct"] = (data["close"] - data["open"]) / data["open"]
    data["high_close_gap"] = data["high"] / data["close"] - 1
    data["close_low_gap"] = data["close"] / data["low"] - 1

    returns = data["return_1d"]
    for window in [5, 10, 20]:
        data[f"rolling_vol_{window}d"] = returns.rolling(window).std()
        data[f"rolling_mean_return_{window}d"] = returns.rolling(window).mean()

    for window in [5, 10, 20, 50]:
        moving_average = data["close"].rolling(window).mean()
        data[f"ma_{window}"] = moving_average
        data[f"ma_{window}_gap"] = data["close"] / moving_average - 1

    rolling_low_20 = data["close"].rolling(20).min()
    rolling_high_20 = data["close"].rolling(20).max()
    denominator = (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    data["position_20d"] = (data["close"] - rolling_low_20) / denominator
    data["drawdown_20d"] = data["close"] / rolling_high_20 - 1

    value_clean = data["value"].replace(0, np.nan)
    data["value_change_1d"] = value_clean / value_clean.shift(1) - 1
    data["log_value"] = np.log(value_clean)

    data["rsi_14"] = calculate_rsi(data["close"], period=14)
    return data


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)

    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi
