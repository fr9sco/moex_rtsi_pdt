from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    classification_metrics,
    last_direction_baseline,
    majority_baseline,
)
from src.features.technical import get_default_feature_columns
from src.models.pdt import PermutationDecisionTreeClassifier
from src.pipelines.supervised import make_walk_forward_splits, split_dataset


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run PDT walk-forward validation")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables_dir = ROOT / "reports" / "tables"
    figures_dir = ROOT / "reports" / "figures"
    trees_dir = ROOT / "reports" / "trees"
    for directory in [tables_dir, figures_dir, trees_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    available_features = set(get_default_feature_columns())
    feature_columns = [feature for feature in PDT_FEATURE_COLUMNS if feature in available_features]
    pd.DataFrame({"feature": feature_columns}).to_csv(
        tables_dir / "stage3_pdt_feature_list.csv",
        index=False,
    )
    all_walk_forward = []
    best_params_rows = []
    test_metric_rows = []
    importance_rows = []
    param_grid = build_pdt_param_grid()
    print(f"Expanded PDT walk-forward grid size: {len(param_grid)} configs per task", flush=True)

    for horizon in args.horizons:
        dataset_path = ROOT / "data" / "processed" / f"moex_supervised_h{horizon}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Run scripts/stage3_build_datasets.py first: missing {dataset_path}"
            )
        dataset = pd.read_csv(dataset_path, parse_dates=["date", "future_date"])

        for secid, security_dataset in dataset.groupby("secid", sort=True):
            print(f"Running PDT pipeline: secid={secid}, horizon={horizon}", flush=True)
            splits = split_dataset(security_dataset)
            train_validation = splits.train_validation.sort_values("date").reset_index(drop=True)
            test = splits.test.sort_values("date").reset_index(drop=True)

            best_params, wf_table = tune_pdt_walk_forward(
                train_validation,
                feature_columns,
                param_grid,
                horizon=horizon,
                secid=secid,
            )
            all_walk_forward.append(wf_table)
            best_params_rows.append(
                {
                    "horizon": horizon,
                    "secid": secid,
                    "selection_method": "walk_forward_validation_expanded_grid",
                    **best_params,
                }
            )

            final_model = fit_pdt(train_validation, feature_columns, best_params)
            pdt_pred = final_model.predict(test[feature_columns])
            majority_pred = majority_baseline(train_validation["target_up"], len(test))
            last_dir_pred = last_direction_baseline(test)

            for model_name, prediction in [
                ("PDT", pdt_pred),
                ("majority_baseline", majority_pred),
                ("last_direction_baseline", last_dir_pred),
            ]:
                metrics = classification_metrics(test["target_up"], prediction)
                test_metric_rows.append(
                    {
                        "horizon": horizon,
                        "secid": secid,
                        "model": model_name,
                        "rows": len(test),
                        "test_first_date": test["date"].min().date().isoformat(),
                        "test_last_date": test["date"].max().date().isoformat(),
                        **metrics,
                    }
                )

            for feature, importance in zip(feature_columns, final_model.feature_importances_):
                importance_rows.append(
                    {
                        "horizon": horizon,
                        "secid": secid,
                        "feature": feature,
                        "importance": importance,
                    }
                )

            tree_path = trees_dir / f"stage3_pdt_h{horizon}_{secid.lower()}.txt"
            tree_path.write_text(final_model.export_text(), encoding="utf-8")

    walk_forward = pd.concat(all_walk_forward, ignore_index=True)
    best_params = pd.DataFrame(best_params_rows)
    test_metrics = pd.DataFrame(test_metric_rows)
    feature_importance = pd.DataFrame(importance_rows)

    walk_forward.to_csv(tables_dir / "stage3_pdt_walk_forward.csv", index=False)
    best_params.to_csv(tables_dir / "stage3_pdt_best_params.csv", index=False)
    test_metrics.to_csv(tables_dir / "stage3_pdt_test_metrics.csv", index=False)
    feature_importance.to_csv(tables_dir / "stage3_pdt_feature_importance.csv", index=False)

    create_figures(test_metrics, feature_importance, figures_dir)
    print(test_metrics.to_string(index=False))
    print(f"Saved Stage 3 PDT tables -> {tables_dir}")
    print(f"Saved Stage 3 PDT figures -> {figures_dir}")


def tune_pdt_walk_forward(
    frame: pd.DataFrame,
    feature_columns: list[str],
    param_grid: list[dict[str, int | bool]],
    horizon: int,
    secid: str,
) -> tuple[dict[str, int | float | bool], pd.DataFrame]:
    folds = make_walk_forward_splits(
        frame,
        min_train_size=1000,
        validation_size=252,
        step_size=504,
    )
    if not folds:
        raise RuntimeError(f"No walk-forward folds for {secid}, h={horizon}")

    rows = []
    for param_id, params in enumerate(param_grid):
        print(
            f"  param_id={param_id + 1}/{len(param_grid)}, h={horizon}, secid={secid}, params={params}",
            flush=True,
        )
        fold_accuracy_scores = []
        fold_balanced_scores = []
        for fold_id, (train_idx, validation_idx) in enumerate(folds):
            train = frame.loc[train_idx]
            validation = frame.loc[validation_idx]
            model = fit_pdt(train, feature_columns, params)
            prediction = model.predict(validation[feature_columns])
            metrics = classification_metrics(validation["target_up"], prediction)
            fold_accuracy_scores.append(metrics["accuracy"])
            fold_balanced_scores.append(metrics["balanced_accuracy"])
            rows.append(
                {
                    "horizon": horizon,
                    "secid": secid,
                    "param_id": param_id,
                    "fold_id": fold_id,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "validation_first_date": validation["date"].min().date().isoformat(),
                    "validation_last_date": validation["date"].max().date().isoformat(),
                    **params,
                    **metrics,
                }
            )

        rows.append(
            {
                "horizon": horizon,
                "secid": secid,
                "param_id": param_id,
                "fold_id": "mean",
                "train_rows": None,
                "validation_rows": None,
                "validation_first_date": None,
                "validation_last_date": None,
                **params,
                "accuracy": sum(fold_accuracy_scores) / len(fold_accuracy_scores),
                "balanced_accuracy": sum(fold_balanced_scores) / len(fold_balanced_scores),
            }
        )

    result = pd.DataFrame(rows)
    means = result[result["fold_id"] == "mean"].copy()
    means = means.sort_values(
        ["balanced_accuracy", "accuracy", "max_depth", "min_samples_leaf", "max_thresholds"],
        ascending=[False, False, True, False, True],
    )
    best = means.iloc[0]
    best_params = {
        "param_id": int(best["param_id"]),
        "walk_forward_accuracy": float(best["accuracy"]),
        "walk_forward_balanced_accuracy": float(best["balanced_accuracy"]),
        "max_depth": int(best["max_depth"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_thresholds": int(best["max_thresholds"]),
        "normalize_etc": bool(best["normalize_etc"]),
    }
    return best_params, result


def build_pdt_param_grid() -> list[dict[str, int | bool]]:
    grid: list[dict[str, int | bool]] = []
    for max_depth, min_samples_leaf, max_thresholds in product(
        [1, 2, 3, 4, 5, 6],
        [10, 20, 50, 100],
        [8, 16],
    ):
        grid.append(
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_thresholds": max_thresholds,
                "normalize_etc": False,
            }
        )

    # проверяем нормированный ETC только на validation
    for max_depth, min_samples_leaf in product([2, 3], [20, 50]):
        grid.append(
            {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_thresholds": 16,
                "normalize_etc": True,
            }
        )
    return grid


def fit_pdt(
    train: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, int | float | bool],
) -> PermutationDecisionTreeClassifier:
    model = PermutationDecisionTreeClassifier(
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        min_samples_split=int(params["min_samples_leaf"]) * 2,
        max_thresholds=int(params["max_thresholds"]),
        normalize_etc=bool(params.get("normalize_etc", False)),
    )
    model.fit(train[feature_columns], train["target_up"])
    return model


def create_figures(
    test_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_data = test_metrics.copy()
    plot_data["task"] = plot_data["secid"] + ", h=" + plot_data["horizon"].astype(str)
    sns.barplot(data=plot_data, x="task", y="accuracy", hue="model", ax=ax)
    ax.set_title("Stage 3 test directional accuracy")
    ax.set_ylim(0.35, 0.65)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(figures_dir / "stage3_pdt_test_accuracy.png", dpi=180)
    plt.close(fig)

    pdt_importance = feature_importance.copy()
    pdt_importance = pdt_importance[pdt_importance["importance"] > 0]
    if pdt_importance.empty:
        return

    top_rows = []
    for (horizon, secid), group in pdt_importance.groupby(["horizon", "secid"], sort=True):
        top = group.sort_values("importance", ascending=False).head(8).copy()
        top["task"] = secid + ", h=" + top["horizon"].astype(str)
        top_rows.append(top)
    top_features = pd.concat(top_rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top_features, x="importance", y="feature", hue="task", ax=ax)
    ax.set_title("Stage 3 PDT feature importance, top splits")
    ax.set_xlabel("Normalized ETC gain importance")
    ax.set_ylabel("")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(figures_dir / "stage3_pdt_feature_importance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
