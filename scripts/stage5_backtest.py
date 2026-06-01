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
    trades = make_strategy_trades(predictions, cost_bps=args.cost_bps)
    metrics = summarize_backtest(trades)

    trades.to_csv(tables_dir / "stage5_backtest_equity.csv", index=False)
    metrics.to_csv(tables_dir / "stage5_backtest_metrics.csv", index=False)
    create_figures(trades, metrics, figures_dir)

    print(metrics.to_string(index=False))
    print(f"Saved Stage 5 tables -> {tables_dir}")
    print(f"Saved Stage 5 figures -> {figures_dir}")


def create_figures(trades: pd.DataFrame, metrics: pd.DataFrame, figures_dir: Path) -> None:
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


if __name__ == "__main__":
    main()
