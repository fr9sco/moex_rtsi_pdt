from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.backtest import make_strategy_trades, summarize_backtest


PREDICTIONS_PATH = ROOT / "reports" / "tables" / "stage4_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run a simple direction backtest")
    parser.add_argument("--predictions", default=str(PREDICTIONS_PATH))
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--min-signal-bps", type=float, default=0.0)
    parser.add_argument("--sensitivity-costs", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0])
    parser.add_argument("--sensitivity-thresholds", type=float, nargs="+", default=[0.0, 10.0, 25.0, 50.0, 100.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Run scripts/stage4_model_comparison.py first: missing {predictions_path}"
        )

    tables_dir = ROOT / "reports" / "tables"
    figures_dir = ROOT / "reports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path, parse_dates=["date", "future_date"])
    trades = make_strategy_trades(
        predictions,
        cost_bps=args.cost_bps,
        min_signal_bps=args.min_signal_bps,
    )
    metrics = summarize_backtest(trades)
    sensitivity = make_sensitivity_table(
        predictions,
        costs=args.sensitivity_costs,
        thresholds=args.sensitivity_thresholds,
    )

    trades.to_csv(tables_dir / "stage5_backtest_equity.csv", index=False)
    metrics.to_csv(tables_dir / "stage5_backtest_metrics.csv", index=False)
    sensitivity.to_csv(tables_dir / "stage5_backtest_sensitivity.csv", index=False)
    create_figures(trades, metrics, sensitivity, figures_dir)

    print(metrics.to_string(index=False))
    print(f"Saved Stage 5 tables -> {tables_dir}")
    print(f"Saved Stage 5 figures -> {figures_dir}")


def make_sensitivity_table(predictions: pd.DataFrame, costs: list[float], thresholds: list[float]) -> pd.DataFrame:
    frames = []
    for cost_bps in costs:
        for min_signal_bps in thresholds:
            trades = make_strategy_trades(
                predictions,
                cost_bps=cost_bps,
                min_signal_bps=min_signal_bps,
            )
            frames.append(summarize_backtest(trades))

    return pd.concat(frames, ignore_index=True)


def create_figures(
    trades: pd.DataFrame,
    metrics: pd.DataFrame,
    sensitivity: pd.DataFrame,
    figures_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid")

    selected_models = [
        "PDT_direction_median_return",
        "LastDailyReturn",
        "LastHorizonReturn",
        "XGBoostRegressor",
        "LSTMRegressor",
    ]
    plot_trades = trades[trades["model"].isin(selected_models)].copy()
    plot_trades["task"] = plot_trades["secid"] + ", h=" + plot_trades["horizon"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for ax, (task, group) in zip(axes.ravel(), plot_trades.groupby("task", sort=True)):
        for model, model_frame in group.groupby("model", sort=True):
            ax.plot(model_frame["date"], model_frame["strategy_equity"], label=model, linewidth=1.0)

        buy_hold = group.drop_duplicates("date").sort_values("date")
        ax.plot(
            buy_hold["date"],
            buy_hold["buy_hold_equity"],
            label="BuyHold",
            color="black",
            linewidth=1.2,
            linestyle="--",
        )
        ax.set_title(task)
        ax.set_ylabel("Equity")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if axes.ravel()[0].get_legend():
        axes.ravel()[0].get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncols=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(figures_dir / "stage5_equity_curves.png", dpi=180)
    plt.close(fig)

    plot_metrics = metrics[metrics["model"].isin(selected_models)].copy()
    plot_metrics["task"] = plot_metrics["secid"] + ", h=" + plot_metrics["horizon"].astype(str)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_metrics, x="task", y="strategy_total_return", hue="model", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Stage 5 strategy total return")
    ax.set_xlabel("")
    ax.set_ylabel("Total return")
    ax.legend(title="", ncols=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "stage5_total_return.png", dpi=180)
    plt.close(fig)

    plot_sensitivity = sensitivity[sensitivity["model"].isin(selected_models)].copy()
    plot_sensitivity["task"] = (
        plot_sensitivity["secid"] + ", h=" + plot_sensitivity["horizon"].astype(str)
    )

    commission = plot_sensitivity[plot_sensitivity["min_signal_bps"] == 0].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for index, (ax, (task, group)) in enumerate(zip(axes.ravel(), commission.groupby("task", sort=True))):
        sns.lineplot(
            data=group,
            x="cost_bps",
            y="strategy_total_return",
            hue="model",
            marker="o",
            ax=ax,
            legend="full" if index == 0 else False,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel("Commission, bps")
        ax.set_ylabel("Total return")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if axes.ravel()[0].get_legend():
        axes.ravel()[0].get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncols=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(figures_dir / "stage5_commission_sensitivity.png", dpi=180)
    plt.close(fig)

    threshold = plot_sensitivity[plot_sensitivity["cost_bps"] == 5].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for index, (ax, (task, group)) in enumerate(zip(axes.ravel(), threshold.groupby("task", sort=True))):
        sns.lineplot(
            data=group,
            x="min_signal_bps",
            y="strategy_total_return",
            hue="model",
            marker="o",
            ax=ax,
            legend="full" if index == 0 else False,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(task)
        ax.set_xlabel("Min signal, bps")
        ax.set_ylabel("Total return")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(figures_dir / "stage5_confidence_threshold.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
