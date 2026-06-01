import unittest

import pandas as pd

from src.evaluation.backtest import describe_returns, make_strategy_trades


class TestStage5Backtest(unittest.TestCase):
    def test_backtest_uses_non_overlapping_rows(self):
        predictions = pd.DataFrame(
            {
                "horizon": [2, 2, 2, 2],
                "secid": ["T"] * 4,
                "model": ["M"] * 4,
                "date": pd.date_range("2024-01-01", periods=4),
                "future_date": pd.date_range("2024-01-03", periods=4),
                "future_return": [0.10, 0.20, -0.05, 0.30],
                "predicted_up": [1, 1, 0, 1],
            }
        )

        trades = make_strategy_trades(predictions, cost_bps=0)

        self.assertEqual(len(trades), 2)
        self.assertEqual(trades["date"].dt.day.tolist(), [1, 3])
        self.assertAlmostEqual(trades["strategy_return"].iloc[0], 0.10)
        self.assertAlmostEqual(trades["strategy_return"].iloc[1], 0.0)

    def test_backtest_charges_position_changes(self):
        predictions = pd.DataFrame(
            {
                "horizon": [1, 1, 1],
                "secid": ["T"] * 3,
                "model": ["M"] * 3,
                "date": pd.date_range("2024-01-01", periods=3),
                "future_date": pd.date_range("2024-01-02", periods=3),
                "future_return": [0.10, 0.10, 0.10],
                "predicted_up": [1, 1, 0],
            }
        )

        trades = make_strategy_trades(predictions, cost_bps=10)

        self.assertAlmostEqual(trades["strategy_return"].iloc[0], 0.099)
        self.assertAlmostEqual(trades["strategy_return"].iloc[1], 0.10)
        self.assertAlmostEqual(trades["strategy_return"].iloc[2], -0.001)

    def test_describe_returns_has_drawdown(self):
        metrics = describe_returns(pd.Series([0.10, -0.20, 0.05]), horizon=1)

        self.assertLess(metrics["max_drawdown"], 0)
        self.assertGreater(metrics["volatility"], 0)


if __name__ == "__main__":
    unittest.main()
