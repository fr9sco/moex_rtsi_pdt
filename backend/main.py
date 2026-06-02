from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


ROOT_DIR = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT_DIR / "reports" / "tables"
TREES_DIR = ROOT_DIR / "reports" / "trees"

app = FastAPI(title="MOEX RTSI PDT dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_csv(name):
    path = TABLES_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {name}")
    return pd.read_csv(path)


def clean_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def rows_to_dicts(df):
    return [
        {column: clean_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def task_filter(df, horizon, secid, model=None):
    result = df[(df["horizon"] == horizon) & (df["secid"] == secid)]
    if model:
        result = result[result["model"] == model]
    return result.copy()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    comparison = read_csv("stage4_model_comparison.csv")
    backtest = read_csv("stage5_backtest_metrics.csv")
    pdt_params = read_csv("stage3_pdt_best_params.csv")

    best_accuracy = comparison.sort_values(
        ["balanced_accuracy", "accuracy"], ascending=False
    ).iloc[0]
    best_rmse = comparison.sort_values("rmse").iloc[0]
    best_return = backtest.sort_values(
        ["strategy_total_return", "strategy_sharpe"], ascending=False
    ).iloc[0]
    pdt_rtsi = comparison[
        (comparison["horizon"] == 1)
        & (comparison["secid"] == "RTSI")
        & (comparison["model"] == "PDT_direction_median_return")
    ].iloc[0]

    return {
        "best_accuracy": {k: clean_value(v) for k, v in best_accuracy.items()},
        "best_rmse": {k: clean_value(v) for k, v in best_rmse.items()},
        "best_return": {k: clean_value(v) for k, v in best_return.items()},
        "pdt_rtsi_h1": {k: clean_value(v) for k, v in pdt_rtsi.items()},
        "pdt_params": rows_to_dicts(pdt_params),
    }


@app.get("/api/model-comparison")
def model_comparison(
    horizon: Optional[int] = Query(default=None, ge=1),
    secid: Optional[str] = None,
):
    df = read_csv("stage4_model_comparison.csv")
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    if secid:
        df = df[df["secid"] == secid]
    df = df.sort_values(["horizon", "secid", "balanced_accuracy"], ascending=[True, True, False])
    return rows_to_dicts(df)


@app.get("/api/backtest")
def backtest(
    horizon: Optional[int] = Query(default=None, ge=1),
    secid: Optional[str] = None,
):
    df = read_csv("stage5_backtest_metrics.csv")
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    if secid:
        df = df[df["secid"] == secid]
    df = df.sort_values(["horizon", "secid", "strategy_total_return"], ascending=[True, True, False])
    return rows_to_dicts(df)


@app.get("/api/predictions")
def predictions(
    horizon: int = Query(default=1, ge=1),
    secid: str = "RTSI",
    model: str = "PDT_direction_median_return",
):
    df = task_filter(read_csv("stage4_predictions.csv"), horizon, secid, model)
    if df.empty:
        raise HTTPException(status_code=404, detail="predictions not found")

    columns = [
        "date",
        "future_date",
        "close",
        "future_close",
        "predicted_close",
        "future_return",
        "predicted_return",
        "target_up",
        "predicted_up",
    ]
    return rows_to_dicts(df[columns])


@app.get("/api/equity")
def equity(
    horizon: int = Query(default=1, ge=1),
    secid: str = "RTSI",
    model: str = "PDT_direction_median_return",
):
    df = task_filter(read_csv("stage5_backtest_equity.csv"), horizon, secid, model)
    if df.empty:
        raise HTTPException(status_code=404, detail="equity not found")

    columns = [
        "date",
        "future_date",
        "position",
        "strategy_return",
        "buy_hold_return",
        "strategy_equity",
        "buy_hold_equity",
    ]
    return rows_to_dicts(df[columns])


@app.get("/api/feature-importance")
def feature_importance(
    source: str = Query(default="pdt", pattern="^(pdt|xgboost)$"),
    horizon: int = Query(default=1, ge=1),
    secid: str = "RTSI",
):
    name = "stage3_pdt_feature_importance.csv"
    if source == "xgboost":
        name = "stage4_xgboost_feature_importance.csv"
    df = task_filter(read_csv(name), horizon, secid)
    if df.empty:
        raise HTTPException(status_code=404, detail="feature importance not found")
    df = df.sort_values("importance", ascending=False).head(12)
    return rows_to_dicts(df)


@app.get("/api/pdt-tree")
def pdt_tree(
    horizon: int = Query(default=1, ge=1),
    secid: str = "RTSI",
):
    path = TREES_DIR / f"stage3_pdt_h{horizon}_{secid.lower()}.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="tree not found")
    return {"text": path.read_text(encoding="utf-8")}
