from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.features.technical import add_technical_features, get_default_feature_columns


TRAIN_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    train_validation: pd.DataFrame
    test: pd.DataFrame


def make_supervised_dataset(
    panel: pd.DataFrame,
    horizon: int,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be a positive integer")

    feature_columns = feature_columns or get_default_feature_columns()
    frames = []
    for secid, frame in panel.groupby("secid", sort=True):
        enriched = add_technical_features(frame)
        enriched["horizon"] = horizon
        # сдвигаем таргет, чтобы не подсматривать в будущее
        enriched["future_date"] = enriched["date"].shift(-horizon)
        enriched["future_close"] = enriched["close"].shift(-horizon)
        enriched["future_return"] = enriched["future_close"] / enriched["close"] - 1
        enriched["target_up"] = (enriched["future_return"] > 0).astype(int)
        enriched["source_secid"] = secid

        required = feature_columns + ["future_date", "future_close", "future_return", "target_up"]
        supervised = enriched.dropna(subset=required).copy()
        frames.append(supervised)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["secid", "date"]).reset_index(drop=True)
    return dataset


def split_dataset(dataset: pd.DataFrame) -> SplitFrames:
    data = dataset.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["future_date"] = pd.to_datetime(data["future_date"])

    train = data[(data["date"] <= TRAIN_END) & (data["future_date"] <= TRAIN_END)].copy()
    validation = data[
        (data["date"] >= VALIDATION_START)
        & (data["date"] <= VALIDATION_END)
        & (data["future_date"] <= VALIDATION_END)
    ].copy()
    train_validation = data[
        (data["date"] <= VALIDATION_END)
        & (data["future_date"] <= VALIDATION_END)
    ].copy()
    test = data[(data["date"] >= TEST_START)].copy()

    return SplitFrames(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        train_validation=train_validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def make_walk_forward_splits(
    frame: pd.DataFrame,
    min_train_size: int = 900,
    validation_size: int = 252,
    step_size: int = 252,
) -> list[tuple[pd.Index, pd.Index]]:
    data = frame.sort_values("date").reset_index(drop=True).copy()
    folds: list[tuple[pd.Index, pd.Index]] = []
    train_end_pos = min_train_size - 1

    while train_end_pos + 1 < len(data):
        validation_start_pos = train_end_pos + 1
        validation_end_pos = min(validation_start_pos + validation_size - 1, len(data) - 1)

        if validation_end_pos - validation_start_pos + 1 < max(30, validation_size // 2):
            break

        train_end_date = data.loc[train_end_pos, "date"]
        validation_start_date = data.loc[validation_start_pos, "date"]
        validation_end_date = data.loc[validation_end_pos, "date"]

        # train-таргет не должен заходить в validation
        train_mask = (data["date"] <= train_end_date) & (data["future_date"] <= train_end_date)
        validation_mask = (
            (data["date"] >= validation_start_date)
            & (data["date"] <= validation_end_date)
            & (data["future_date"] <= validation_end_date)
        )

        train_idx = data.index[train_mask]
        validation_idx = data.index[validation_mask]
        if len(train_idx) >= min_train_size // 2 and len(validation_idx) > 0:
            folds.append((train_idx, validation_idx))

        train_end_pos += step_size

    return folds


def summarize_splits(dataset: pd.DataFrame) -> pd.DataFrame:
    splits = split_dataset(dataset)
    rows = []
    for split_name in ["train", "validation", "train_validation", "test"]:
        split_frame = getattr(splits, split_name)
        for secid, frame in split_frame.groupby("secid", sort=True):
            rows.append(
                {
                    "horizon": int(frame["horizon"].iloc[0]),
                    "split": split_name,
                    "secid": secid,
                    "rows": len(frame),
                    "first_date": frame["date"].min().date().isoformat(),
                    "last_date": frame["date"].max().date().isoformat(),
                    "first_future_date": frame["future_date"].min().date().isoformat(),
                    "last_future_date": frame["future_date"].max().date().isoformat(),
                    "positive_share": frame["target_up"].mean(),
                }
            )
    return pd.DataFrame(rows)
