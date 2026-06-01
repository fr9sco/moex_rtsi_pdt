import unittest

import numpy as np
import pandas as pd

from src.evaluation.metrics import combined_forecast_metrics, regression_metrics
from src.pipelines.sequences import build_lstm_sequences


class TestStage4MetricsAndSequences(unittest.TestCase):
    def test_regression_metrics_basic(self):
        metrics = regression_metrics([100, 200], [110, 190])
        self.assertAlmostEqual(metrics["mae"], 10.0)
        self.assertAlmostEqual(metrics["rmse"], 10.0)
        self.assertAlmostEqual(metrics["mape_pct"], 7.5)

    def test_combined_forecast_metrics_direction(self):
        frame = pd.DataFrame(
            {
                "future_close": [110.0, 90.0],
                "target_up": [1, 0],
            }
        )
        metrics = combined_forecast_metrics(
            frame,
            predicted_close=np.array([108.0, 95.0]),
            predicted_return=np.array([0.02, -0.01]),
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["mae"], 0)

    def test_lstm_sequences_use_past_window_ending_at_target_row(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=5),
                "x": [1, 2, 3, 4, 5],
                "target": [10, 20, 30, 40, 50],
            }
        )
        X, y, positions = build_lstm_sequences(
            frame,
            feature_columns=["x"],
            target_column="target",
            lookback=3,
            eligible_positions={3, 4},
        )
        self.assertEqual(X.shape, (2, 3, 1))
        np.testing.assert_array_equal(X[0, :, 0], np.array([2, 3, 4]))
        np.testing.assert_array_equal(y, np.array([40, 50], dtype=np.float32))
        np.testing.assert_array_equal(positions, np.array([3, 4]))


if __name__ == "__main__":
    unittest.main()

