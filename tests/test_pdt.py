import unittest

import numpy as np
import pandas as pd

from src.models.pdt import (
    PermutationDecisionTreeClassifier,
    effort_to_compress,
    etc_gain,
)


class TestEffortToCompress(unittest.TestCase):
    def test_constant_sequence_has_zero_etc(self):
        self.assertEqual(effort_to_compress([1, 1, 1, 1]), 0.0)

    def test_non_constant_sequence_has_positive_etc(self):
        value = effort_to_compress([0, 1, 0, 1, 0, 1], normalized=True)
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_raw_etc_not_normalized(self):
        raw = effort_to_compress([0, 1, 0, 1], normalized=False)
        normalized = effort_to_compress([0, 1, 0, 1], normalized=True)
        self.assertGreater(raw, normalized)

    def test_split_gain_is_positive_for_clean_split(self):
        parent = [0, 0, 0, 1, 1, 1]
        left = [0, 0, 0]
        right = [1, 1, 1]
        self.assertGreater(etc_gain(parent, left, right), 0.0)


class TestPermutationDecisionTreeClassifier(unittest.TestCase):
    def test_tree_fits_simple_rule(self):
        X = pd.DataFrame(
            {
                "signal": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
                "noise": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
            }
        )
        y = np.array([0, 0, 0, 1, 1, 1])

        model = PermutationDecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
        model.fit(X, y)

        np.testing.assert_array_equal(model.predict(X), y)
        self.assertIn("signal", model.export_text())
        self.assertGreaterEqual(model.get_n_leaves(), 2)

    def test_predict_proba_rows_sum_to_one(self):
        X = pd.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]})
        y = np.array([0, 0, 1, 1])

        model = PermutationDecisionTreeClassifier(max_depth=1)
        model.fit(X, y)
        proba = model.predict_proba(X)

        self.assertEqual(proba.shape, (4, 2))
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))

    def test_raises_on_nan(self):
        X = pd.DataFrame({"x": [0.0, np.nan, 1.0]})
        y = np.array([0, 0, 1])

        model = PermutationDecisionTreeClassifier()
        with self.assertRaises(ValueError):
            model.fit(X, y)


if __name__ == "__main__":
    unittest.main()
