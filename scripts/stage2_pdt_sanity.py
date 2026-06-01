from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pdt import PermutationDecisionTreeClassifier, effort_to_compress


DATA_PATH = ROOT / "data" / "processed" / "moex_indices_daily_2015-01-05_2026-05-27.csv"
TABLES_DIR = ROOT / "reports" / "tables"
TREES_DIR = ROOT / "reports" / "trees"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TREES_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    results = []

    for secid, frame in data.groupby("secid", sort=True):
        dataset = make_small_dataset(frame)
        train, validation = chronological_split(dataset)

        feature_columns = [
            "return_1d",
            "return_5d",
            "range_pct",
            "gap_pct",
            "ma_5_gap",
            "ma_20_gap",
        ]

        model = PermutationDecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=25,
            min_samples_split=50,
            max_thresholds=32,
        )
        model.fit(train[feature_columns], train["target_up"])
        prediction = model.predict(validation[feature_columns])

        majority_class = int(train["target_up"].mode().iloc[0])
        majority_prediction = np.full(len(validation), majority_class)

        accuracy = float((prediction == validation["target_up"].to_numpy()).mean())
        majority_accuracy = float(
            (majority_prediction == validation["target_up"].to_numpy()).mean()
        )

        tree_path = TREES_DIR / f"stage2_pdt_tree_{secid.lower()}.txt"
        tree_path.write_text(model.export_text(), encoding="utf-8")

        importances = dict(zip(feature_columns, model.feature_importances_))
        best_feature = max(importances, key=importances.get)
        results.append(
            {
                "secid": secid,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_first_date": train["date"].min().date().isoformat(),
                "train_last_date": train["date"].max().date().isoformat(),
                "validation_first_date": validation["date"].min().date().isoformat(),
                "validation_last_date": validation["date"].max().date().isoformat(),
                "pdt_directional_accuracy": accuracy,
                "majority_baseline_accuracy": majority_accuracy,
                "tree_depth": model.get_depth(),
                "n_leaves": model.get_n_leaves(),
                "root_etc_raw": effort_to_compress(train["target_up"], normalized=False),
                "root_etc_normalized": effort_to_compress(train["target_up"], normalized=True),
                "best_feature_by_gain": best_feature,
                "best_feature_importance": importances[best_feature],
            }
        )

    result_frame = pd.DataFrame(results)
    output_path = TABLES_DIR / "stage2_pdt_sanity.csv"
    result_frame.to_csv(output_path, index=False)
    print(result_frame.to_string(index=False))
    print(f"Saved sanity table: {output_path}")
    print(f"Saved tree exports: {TREES_DIR}")


def make_small_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values("date").copy()
    frame["future_close"] = frame["close"].shift(-1)
    frame["future_date"] = frame["date"].shift(-1)
    frame["target_up"] = (frame["future_close"] > frame["close"]).astype(int)

    frame["return_5d"] = frame["close"].pct_change(5)
    frame["ma_5"] = frame["close"].rolling(5).mean()
    frame["ma_20"] = frame["close"].rolling(20).mean()
    frame["ma_5_gap"] = frame["close"] / frame["ma_5"] - 1
    frame["ma_20_gap"] = frame["close"] / frame["ma_20"] - 1

    feature_columns = [
        "return_1d",
        "return_5d",
        "range_pct",
        "gap_pct",
        "ma_5_gap",
        "ma_20_gap",
    ]
    return frame.dropna(subset=feature_columns + ["future_close"]).reset_index(drop=True)


def chronological_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp("2021-12-31")
    validation_start = pd.Timestamp("2022-01-01")
    validation_end = pd.Timestamp("2023-12-31")

    train = dataset[
        (dataset["date"] <= train_end)
        & (dataset["future_date"] <= train_end)
    ].copy()
    validation = dataset[
        (dataset["date"] >= validation_start)
        & (dataset["date"] <= validation_end)
        & (dataset["future_date"] <= validation_end)
    ].copy()

    return train, validation


if __name__ == "__main__":
    main()
