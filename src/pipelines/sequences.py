from __future__ import annotations

import numpy as np
import pandas as pd


def build_lstm_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
    eligible_positions: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError("lookback must be positive")

    data = frame.sort_values("date").reset_index(drop=True)
    values = data[feature_columns].to_numpy(dtype=np.float32)
    targets = data[target_column].to_numpy(dtype=np.float32)
    positions = []
    sequences = []
    y = []

    for pos in range(lookback - 1, len(data)):
        if eligible_positions is not None and pos not in eligible_positions:
            continue
        # окно может смотреть назад, но таргет остается в своем split
        window = values[pos - lookback + 1 : pos + 1]
        if not np.isfinite(window).all() or not np.isfinite(targets[pos]):
            continue
        sequences.append(window)
        y.append(targets[pos])
        positions.append(pos)

    if not sequences:
        return (
            np.empty((0, lookback, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=int),
        )
    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(positions, dtype=int),
    )
