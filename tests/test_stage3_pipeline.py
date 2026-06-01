import unittest

import pandas as pd

from src.features.technical import get_default_feature_columns
from src.pipelines.supervised import make_supervised_dataset, split_dataset


class TestStage3Pipeline(unittest.TestCase):
    def test_supervised_target_uses_future_close(self):
        panel = pd.DataFrame(
            {
                "secid": ["T"] * 60,
                "date": pd.date_range("2020-01-01", periods=60, freq="D"),
                "open": range(100, 160),
                "high": range(101, 161),
                "low": range(99, 159),
                "close": list(range(100, 130)) + list(range(130, 100, -1)),
                "value": [1000 + i for i in range(60)],
            }
        )

        dataset = make_supervised_dataset(panel, horizon=1)
        row = dataset.iloc[0]
        source_pos = panel.index[panel["date"] == row["date"]][0]

        self.assertEqual(row["future_close"], panel.loc[source_pos + 1, "close"])
        self.assertEqual(
            row["target_up"],
            int(panel.loc[source_pos + 1, "close"] > panel.loc[source_pos, "close"]),
        )

    def test_feature_columns_do_not_include_target_columns(self):
        forbidden = {"future_close", "future_return", "future_date", "target_up"}
        self.assertTrue(forbidden.isdisjoint(get_default_feature_columns()))

    def test_split_prevents_train_label_crossing_boundary(self):
        panel = pd.DataFrame(
            {
                "secid": ["T"] * 2600,
                "date": pd.bdate_range("2015-01-01", periods=2600),
                "open": [100.0 + i * 0.01 for i in range(2600)],
                "high": [101.0 + i * 0.01 for i in range(2600)],
                "low": [99.0 + i * 0.01 for i in range(2600)],
                "close": [100.0 + i * 0.01 for i in range(2600)],
                "value": [1000.0 + i for i in range(2600)],
            }
        )
        dataset = make_supervised_dataset(panel, horizon=5)
        splits = split_dataset(dataset)

        self.assertLessEqual(splits.train["future_date"].max(), pd.Timestamp("2021-12-31"))
        self.assertLessEqual(splits.validation["future_date"].max(), pd.Timestamp("2023-12-31"))
        self.assertGreaterEqual(splits.test["date"].min(), pd.Timestamp("2024-01-01"))


if __name__ == "__main__":
    unittest.main()
