from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.moex_client import MoexHistoryRequest, MoexIssClient, combine_histories


DEFAULT_SECURITIES = ("IMOEX", "RTSI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="download MOEX data and build EDA artifacts")
    parser.add_argument("--from-date", default="2015-01-01")
    parser.add_argument("--till-date", default=date.today().isoformat())
    parser.add_argument("--securities", nargs="+", default=list(DEFAULT_SECURITIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    figures_dir = ROOT / "reports" / "figures"
    tables_dir = ROOT / "reports" / "tables"
    for directory in [raw_dir, processed_dir, figures_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    client = MoexIssClient()
    frames = []
    for secid in args.securities:
        request = MoexHistoryRequest(
            secid=secid,
            from_date=args.from_date,
            till_date=args.till_date,
        )
        frame = client.fetch_index_history(request)
        if frame.empty:
            raise RuntimeError(f"MOEX ISS returned no rows for {secid}")

        raw_path = raw_dir / f"moex_{secid.lower()}_history_{args.from_date}_{args.till_date}.csv"
        frame.to_csv(raw_path, index=False)
        frames.append(frame)
        print(f"Saved raw {secid}: {len(frame)} rows -> {raw_path}")

    panel = combine_histories(frames)
    first_date = panel["date"].min().date().isoformat()
    last_date = panel["date"].max().date().isoformat()

    processed_csv = processed_dir / f"moex_indices_daily_{first_date}_{last_date}.csv"
    processed_parquet = processed_dir / f"moex_indices_daily_{first_date}_{last_date}.parquet"
    panel.to_csv(processed_csv, index=False)
    panel.to_parquet(processed_parquet, index=False)
    print(f"Saved processed CSV: {processed_csv}")
    print(f"Saved processed Parquet: {processed_parquet}")

    summary = build_summary(panel)
    summary_path = tables_dir / "stage1_data_summary.csv"
    summary.to_csv(summary_path, index=False)

    quality = build_quality_report(panel)
    quality_path = tables_dir / "stage1_quality_report.csv"
    quality.to_csv(quality_path, index=False)

    returns = build_return_summary(panel)
    returns_path = tables_dir / "stage1_return_summary.csv"
    returns.to_csv(returns_path, index=False)

    gaps = build_gap_report(panel)
    gaps_path = tables_dir / "stage1_date_gaps.csv"
    gaps.to_csv(gaps_path, index=False)

    anomalies = build_anomaly_report(panel)
    anomalies_path = tables_dir / "stage1_anomalies.csv"
    anomalies.to_csv(anomalies_path, index=False)

    drawdowns = build_drawdown_summary(panel)
    drawdowns_path = tables_dir / "stage1_drawdown_summary.csv"
    drawdowns.to_csv(drawdowns_path, index=False)

    correlation = build_return_correlation(panel)
    correlation_path = tables_dir / "stage1_return_correlation.csv"
    correlation.to_csv(correlation_path, index=False)

    split = build_split_recommendation(panel)
    split_path = tables_dir / "stage1_split_recommendation.csv"
    split.to_csv(split_path, index=False)

    create_figures(panel, figures_dir)
    print(f"Saved tables -> {tables_dir}")
    print(f"Saved figures -> {figures_dir}")


def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for secid, frame in panel.groupby("secid", sort=True):
        rows.append(
            {
                "secid": secid,
                "rows": len(frame),
                "first_date": frame["date"].min().date().isoformat(),
                "last_date": frame["date"].max().date().isoformat(),
                "currency": frame["currency"].dropna().iloc[0],
                "first_close": frame["close"].iloc[0],
                "last_close": frame["close"].iloc[-1],
                "min_close": frame["close"].min(),
                "max_close": frame["close"].max(),
                "mean_close": frame["close"].mean(),
                "std_close": frame["close"].std(),
                "mean_return_1d": frame["return_1d"].mean(),
                "std_return_1d": frame["return_1d"].std(),
            }
        )
    return pd.DataFrame(rows)


def build_quality_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    required = ["open", "high", "low", "close", "value"]
    for secid, frame in panel.groupby("secid", sort=True):
        duplicated_dates = int(frame.duplicated("date").sum())
        missing_counts = frame[required].isna().sum().to_dict()
        non_positive_prices = int((frame[["open", "high", "low", "close"]] <= 0).sum().sum())
        ohlc_violations = int(
            (
                (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
                | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            ).sum()
        )
        zero_value_rows = int((frame["value"].fillna(0) <= 0).sum())
        extreme_return_rows = int((frame["return_1d"].abs() > 0.2).sum())
        rows.append(
            {
                "secid": secid,
                "duplicated_dates": duplicated_dates,
                "missing_open": int(missing_counts["open"]),
                "missing_high": int(missing_counts["high"]),
                "missing_low": int(missing_counts["low"]),
                "missing_close": int(missing_counts["close"]),
                "missing_value": int(missing_counts["value"]),
                "non_positive_prices": non_positive_prices,
                "ohlc_violations": ohlc_violations,
                "zero_or_missing_value_rows": zero_value_rows,
                "abs_return_gt_20pct_rows": extreme_return_rows,
            }
        )
    return pd.DataFrame(rows)


def build_return_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for secid, frame in panel.groupby("secid", sort=True):
        returns = frame["return_1d"].dropna()
        rows.append(
            {
                "secid": secid,
                "count": len(returns),
                "mean": returns.mean(),
                "std": returns.std(),
                "min": returns.min(),
                "q01": returns.quantile(0.01),
                "q05": returns.quantile(0.05),
                "median": returns.median(),
                "q95": returns.quantile(0.95),
                "q99": returns.quantile(0.99),
                "max": returns.max(),
                "positive_share": (returns > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_gap_report(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for secid, frame in panel.groupby("secid", sort=True):
        gaps = frame["date"].diff().dt.days.dropna()
        rows.append(
            {
                "secid": secid,
                "min_gap_days": int(gaps.min()),
                "median_gap_days": float(gaps.median()),
                "max_gap_days": int(gaps.max()),
                "gap_gt_7_days_count": int((gaps > 7).sum()),
                "gap_gt_7_days_examples": "; ".join(
                    frame.loc[gaps[gaps > 7].head(5).index, "date"]
                    .dt.date.astype(str)
                    .tolist()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_anomaly_report(panel: pd.DataFrame) -> pd.DataFrame:
    extreme_returns = panel["return_1d"].abs() > 0.2
    zero_value = panel["value"].fillna(0) <= 0
    anomalies = panel.loc[
        extreme_returns | zero_value,
        ["secid", "date", "open", "high", "low", "close", "value", "return_1d", "gap_pct"],
    ].copy()
    anomalies["reason"] = np.select(
        [extreme_returns.loc[anomalies.index], zero_value.loc[anomalies.index]],
        ["abs_return_gt_20pct", "zero_or_missing_value"],
        default="check",
    )
    return anomalies.sort_values(["date", "secid"]).reset_index(drop=True)


def build_drawdown_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for secid, frame in panel.groupby("secid", sort=True):
        close = frame.set_index("date")["close"]
        drawdown = close / close.cummax() - 1
        max_drawdown_date = drawdown.idxmin()
        rows.append(
            {
                "secid": secid,
                "max_drawdown": drawdown.min(),
                "max_drawdown_date": max_drawdown_date.date().isoformat(),
                "peak_before_max_drawdown": close.loc[:max_drawdown_date].cummax().iloc[-1],
                "close_at_max_drawdown": close.loc[max_drawdown_date],
            }
        )
    return pd.DataFrame(rows)


def build_return_correlation(panel: pd.DataFrame) -> pd.DataFrame:
    returns = panel.pivot(index="date", columns="secid", values="return_1d").dropna()
    corr = returns.corr().reset_index().rename(columns={"secid": "row_secid"})
    return corr


def build_split_recommendation(panel: pd.DataFrame) -> pd.DataFrame:
    boundaries = [
        ("train", None, pd.Timestamp("2021-12-31")),
        ("validation", pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
        ("test", pd.Timestamp("2024-01-01"), None),
    ]
    rows = []
    for split_name, start, end in boundaries:
        mask = pd.Series(True, index=panel.index)
        if start is not None:
            mask &= panel["date"] >= start
        if end is not None:
            mask &= panel["date"] <= end
        split_frame = panel[mask]
        for secid, frame in split_frame.groupby("secid", sort=True):
            rows.append(
                {
                    "split": split_name,
                    "secid": secid,
                    "rows": len(frame),
                    "first_date": frame["date"].min().date().isoformat(),
                    "last_date": frame["date"].max().date().isoformat(),
                }
            )
    return pd.DataFrame(rows)


def create_figures(panel: pd.DataFrame, figures_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    close_pivot = panel.pivot(index="date", columns="secid", values="close")
    normalized = close_pivot / close_pivot.iloc[0] * 100

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    close_pivot.plot(ax=axes[0], linewidth=1.3)
    axes[0].set_title("Close values: IMOEX and RTSI")
    axes[0].set_ylabel("Index points")
    axes[0].legend(title="")

    normalized.plot(ax=axes[1], linewidth=1.3)
    axes[1].set_title("Normalized close, first observation = 100")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(title="")
    fig.tight_layout()
    fig.savefig(figures_dir / "stage1_close_and_normalized.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, (secid, frame) in zip(axes, panel.groupby("secid", sort=True)):
        ax.plot(frame["date"], frame["return_1d"], linewidth=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{secid}: daily returns")
        ax.set_ylabel("Return")
    fig.tight_layout()
    fig.savefig(figures_dir / "stage1_daily_returns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (secid, frame) in zip(axes, panel.groupby("secid", sort=True)):
        sns.histplot(frame["return_1d"].dropna(), bins=80, kde=True, ax=ax)
        ax.set_title(f"{secid}: daily return distribution")
        ax.set_xlabel("Daily return")
    fig.tight_layout()
    fig.savefig(figures_dir / "stage1_return_distributions.png", dpi=180)
    plt.close(fig)

    volatility_frames = []
    for secid, frame in panel.groupby("secid", sort=True):
        tmp = frame[["date", "secid", "return_1d"]].copy()
        tmp["rolling_vol_20d"] = tmp["return_1d"].rolling(20).std() * np.sqrt(252)
        volatility_frames.append(tmp)
    volatility = pd.concat(volatility_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(13, 5))
    for secid, frame in volatility.groupby("secid", sort=True):
        ax.plot(frame["date"], frame["rolling_vol_20d"], label=secid, linewidth=1.0)
    ax.set_title("20-day annualized rolling volatility")
    ax.set_ylabel("Annualized volatility")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(figures_dir / "stage1_rolling_volatility_20d.png", dpi=180)
    plt.close(fig)

    merged_returns = panel.pivot(index="date", columns="secid", values="return_1d").dropna()
    if len(merged_returns) > 1:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(merged_returns.corr(), annot=True, cmap="vlag", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Daily return correlation")
        fig.tight_layout()
        fig.savefig(figures_dir / "stage1_return_correlation.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
