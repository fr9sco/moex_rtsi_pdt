from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


def effort_to_compress(sequence: Iterable[Any], normalized: bool = False) -> float:
    seq = _factorize(sequence)
    n = len(seq)
    if n <= 1 or len(set(seq)) <= 1:
        return 0.0

    effort = 0
    next_symbol = max(seq) + 1

    while len(seq) > 1 and len(set(seq)) > 1:
        # сжимаем самую частую соседнюю пару
        pair_counts = Counter(zip(seq[:-1], seq[1:]))
        best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]

        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                new_seq.append(next_symbol)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1

        seq = new_seq
        next_symbol += 1
        effort += 1

    if not normalized:
        return float(effort)
    return effort / (n - 1)


def etc_gain(
    parent: Iterable[Any],
    left: Iterable[Any],
    right: Iterable[Any],
    normalized: bool = False,
) -> float:
    parent = list(parent)
    left = list(left)
    right = list(right)
    n = len(parent)
    if n == 0:
        return 0.0

    parent_impurity = effort_to_compress(parent, normalized=normalized)
    child_impurity = (
        len(left) / n * effort_to_compress(left, normalized=normalized)
        + len(right) / n * effort_to_compress(right, normalized=normalized)
    )
    return parent_impurity - child_impurity


@dataclass
class _Node:
    depth: int
    n_samples: int
    prediction_index: int
    proba: np.ndarray
    impurity: float
    feature_index: int | None = None
    feature_name: str | None = None
    threshold: float | None = None
    gain: float = 0.0
    left: "_Node | None" = None
    right: "_Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class PermutationDecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_thresholds: int | None = 64,
        min_gain: float = 1e-12,
        normalize_etc: bool = False,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_thresholds = max_thresholds
        self.min_gain = min_gain
        self.normalize_etc = normalize_etc

    def fit(self, X: pd.DataFrame | np.ndarray, y: Iterable[Any]) -> "PermutationDecisionTreeClassifier":
        X_array, feature_names = _prepare_X(X)
        y_array = np.asarray(list(y))
        if len(X_array) != len(y_array):
            raise ValueError("X and y must have the same number of rows")
        if len(y_array) == 0:
            raise ValueError("Cannot fit a tree on an empty dataset")
        if not np.isfinite(X_array).all():
            raise ValueError("PDT currently expects finite numeric features without NaN")

        self.feature_names_ = feature_names
        self.classes_, y_encoded = np.unique(y_array, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_array.shape[1]
        self.root_ = self._build_node(X_array, y_encoded, depth=0)
        self.feature_importances_ = self._calculate_feature_importances()
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_array, _ = _prepare_X(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        if not np.isfinite(X_array).all():
            raise ValueError("PDT currently expects finite numeric features without NaN")

        prediction_indices = [self._predict_one(row).prediction_index for row in X_array]
        return self.classes_[prediction_indices]

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X_array, _ = _prepare_X(X)
        if X_array.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        if not np.isfinite(X_array).all():
            raise ValueError("PDT currently expects finite numeric features without NaN")

        return np.vstack([self._predict_one(row).proba for row in X_array])

    def export_text(self, max_depth: int | None = None) -> str:
        self._check_is_fitted()
        lines: list[str] = []
        self._export_node(self.root_, lines, max_depth=max_depth)
        return "\n".join(lines)

    def get_depth(self) -> int:
        self._check_is_fitted()
        return _node_depth(self.root_)

    def get_n_leaves(self) -> int:
        self._check_is_fitted()
        return _count_leaves(self.root_)

    def _build_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        proba = _class_proba(y, self.n_classes_)
        prediction_index = int(np.argmax(proba))
        node = _Node(
            depth=depth,
            n_samples=len(y),
            prediction_index=prediction_index,
            proba=proba,
            impurity=effort_to_compress(y, normalized=self.normalize_etc),
        )

        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return node

        split = self._find_best_split(X, y)
        if split is None or split["gain"] <= self.min_gain:
            return node

        feature_index = split["feature_index"]
        threshold = split["threshold"]
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_index
        node.feature_name = self.feature_names_[feature_index]
        node.threshold = threshold
        node.gain = split["gain"]
        node.left = self._build_node(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_node(X[right_mask], y[right_mask], depth + 1)
        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any] | None:
        best_split: dict[str, Any] | None = None
        parent_impurity = effort_to_compress(y, normalized=self.normalize_etc)
        n_samples = len(y)

        for feature_index in range(X.shape[1]):
            values = X[:, feature_index]
            for threshold in _candidate_thresholds(values, self.max_thresholds):
                left_mask = values <= threshold
                n_left = int(left_mask.sum())
                n_right = len(y) - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_impurity = effort_to_compress(
                    y[left_mask],
                    normalized=self.normalize_etc,
                )
                right_impurity = effort_to_compress(
                    y[~left_mask],
                    normalized=self.normalize_etc,
                )
                gain = parent_impurity - (
                    n_left / n_samples * left_impurity
                    + n_right / n_samples * right_impurity
                )
                if best_split is None or gain > best_split["gain"]:
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": float(threshold),
                        "gain": float(gain),
                    }

        return best_split

    def _predict_one(self, row: np.ndarray) -> _Node:
        node = self.root_
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def _export_node(
        self,
        node: _Node,
        lines: list[str],
        max_depth: int | None,
        prefix: str = "",
    ) -> None:
        class_label = self.classes_[node.prediction_index]
        if node.is_leaf or (max_depth is not None and node.depth >= max_depth):
            lines.append(
                f"{prefix}leaf: class={class_label}, "
                f"samples={node.n_samples}, impurity={node.impurity:.4f}, "
                f"proba={np.round(node.proba, 3).tolist()}"
            )
            return

        lines.append(
            f"{prefix}if {node.feature_name} <= {node.threshold:.6g} "
            f"(gain={node.gain:.4f}, samples={node.n_samples})"
        )
        self._export_node(node.left, lines, max_depth, prefix + "  ")
        lines.append(f"{prefix}else")
        self._export_node(node.right, lines, max_depth, prefix + "  ")

    def _calculate_feature_importances(self) -> np.ndarray:
        importances = np.zeros(self.n_features_in_, dtype=float)

        def walk(node: _Node) -> None:
            if node.is_leaf:
                return
            importances[node.feature_index] += node.gain * node.n_samples
            walk(node.left)
            walk(node.right)

        walk(self.root_)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "root_"):
            raise ValueError("The tree is not fitted yet")


def _factorize(sequence: Iterable[Any]) -> list[int]:
    mapping: dict[Any, int] = {}
    encoded: list[int] = []
    for value in sequence:
        if value not in mapping:
            mapping[value] = len(mapping)
        encoded.append(mapping[value])
    return encoded


def _prepare_X(X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        feature_names = [str(column) for column in X.columns]
        X_array = X.to_numpy(dtype=float)
    else:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        feature_names = [f"x{i}" for i in range(X_array.shape[1])]

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array or DataFrame")
    return X_array, feature_names


def _class_proba(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes)
    return counts / counts.sum()


def _candidate_thresholds(values: np.ndarray, max_thresholds: int | None) -> np.ndarray:
    unique_values = np.unique(values)
    if len(unique_values) <= 1:
        return np.array([], dtype=float)

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    if max_thresholds is None or len(midpoints) <= max_thresholds:
        return midpoints

    positions = np.linspace(0, len(midpoints) - 1, max_thresholds)
    indices = np.unique(np.round(positions).astype(int))
    return midpoints[indices]


def _node_depth(node: _Node) -> int:
    if node.is_leaf:
        return node.depth
    return max(_node_depth(node.left), _node_depth(node.right))


def _count_leaves(node: _Node) -> int:
    if node.is_leaf:
        return 1
    return _count_leaves(node.left) + _count_leaves(node.right)
