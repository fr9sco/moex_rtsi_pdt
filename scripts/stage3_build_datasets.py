from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.technical import get_default_feature_columns
from src.pipelines.supervised import make_supervised_dataset, summarize_splits


INPUT_PATH = ROOT / "data" / "processed" / "moex_indices_daily_2015-01-05_2026-05-27.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build supervised datasets for stage 3")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    processed_dir = ROOT / "data" / "processed"
    tables_dir = ROOT / "reports" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(input_path, parse_dates=["date"])
    feature_columns = get_default_feature_columns()

    all_summaries = []
    dataset_rows = []
    for horizon in args.horizons:
        dataset = make_supervised_dataset(panel, horizon=horizon, feature_columns=feature_columns)
        output_path = processed_dir / f"moex_supervised_h{horizon}.csv"
        dataset.to_csv(output_path, index=False)

        summary = summarize_splits(dataset)
        all_summaries.append(summary)
        for secid, frame in dataset.groupby("secid", sort=True):
            dataset_rows.append(
                {
                    "horizon": horizon,
                    "secid": secid,
                    "rows": len(frame),
                    "first_date": frame["date"].min().date().isoformat(),
                    "last_date": frame["date"].max().date().isoformat(),
                    "positive_share": frame["target_up"].mean(),
                    "feature_count": len(feature_columns),
                }
            )
        print(f"Saved supervised h={horizon}: {len(dataset)} rows -> {output_path}")

    pd.DataFrame(dataset_rows).to_csv(tables_dir / "stage3_dataset_summary.csv", index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(
        tables_dir / "stage3_split_summary.csv",
        index=False,
    )
    pd.DataFrame({"feature": feature_columns}).to_csv(
        tables_dir / "stage3_feature_list.csv",
        index=False,
    )
    print(f"Saved Stage 3 tables -> {tables_dir}")


if __name__ == "__main__":
    main()
