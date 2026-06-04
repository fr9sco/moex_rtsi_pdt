from __future__ import annotations

import numpy as np
import pandas as pd


def make_strategy_trades(
    predictions: pd.DataFrame,
    cost_bps: float = 5.0,
    min_signal_bps: float = 0.0,
) -> pd.DataFrame:
    rows = []
    cost = cost_bps / 10_000

    group_columns = ["horizon", "secid", "model"]
    for (horizon, secid, model), group in predictions.groupby(group_columns, sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        trades = group.iloc[:: int(horizon)].copy()

        if "predicted_return" in trades:
            signal_strength = trades["predicted_return"].astype(float).abs() * 10_000
        elif min_signal_bps > 0:
            raise ValueError("predicted_return is required when min_signal_bps > 0")
        else:
            signal_strength = pd.Series(np.inf, index=trades.index)

        strong_signal = signal_strength >= min_signal_bps
        position = (trades["predicted_up"].astype(int) & strong_signal.astype(int)).astype(int)
        previous_position = position.shift(1, fill_value=0)

        trades["position"] = position
        trades["signal_strength_bps"] = signal_strength
        trades["position_change"] = (position - previous_position).abs()
        trades["transaction_cost"] = trades["position_change"] * cost
        trades["strategy_return"] = position * trades["future_return"] - trades["transaction_cost"]
        trades["buy_hold_return"] = trades["future_return"]
        trades["strategy_equity"] = (1 + trades["strategy_return"]).cumprod()
        trades["buy_hold_equity"] = (1 + trades["buy_hold_return"]).cumprod()
        trades["cost_bps"] = cost_bps
        trades["min_signal_bps"] = min_signal_bps

        rows.append(trades)

    return pd.concat(rows, ignore_index=True)


def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["horizon", "secid", "model", "cost_bps", "min_signal_bps"]
    for (horizon, secid, model, cost_bps, min_signal_bps), group in trades.groupby(group_columns, sort=True):
        strategy = describe_returns(group["strategy_return"], int(horizon))
        buy_hold = describe_returns(group["buy_hold_return"], int(horizon))

        rows.append(
            {
                "horizon": int(horizon),
                "secid": secid,
                "model": model,
                "rows": len(group),
                "first_date": group["date"].min(),
                "last_date": group["date"].max(),
                "cost_bps": float(cost_bps),
                "min_signal_bps": float(min_signal_bps),
                "exposure": float(group["position"].mean()),
                "entries": int(((group["position"] == 1) & (group["position"].shift(1, fill_value=0) == 0)).sum()),
                **{f"strategy_{key}": value for key, value in strategy.items()},
                **{f"buy_hold_{key}": value for key, value in buy_hold.items()},
            }
        )

    return pd.DataFrame(rows)


def describe_returns(returns: pd.Series, horizon: int) -> dict[str, float]:
    returns = returns.astype(float)
    equity = (1 + returns).cumprod()
    annual_factor = 252 / horizon

    total_return = float(equity.iloc[-1] - 1)
    annual_return = float(equity.iloc[-1] ** (annual_factor / len(returns)) - 1)
    volatility = float(returns.std(ddof=0) * np.sqrt(annual_factor))
    sharpe = 0.0 if volatility == 0 else float(returns.mean() / returns.std(ddof=0) * np.sqrt(annual_factor))
    max_drawdown = float((equity / equity.cummax() - 1).min())
    hit_rate = float((returns > 0).mean())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
    }
