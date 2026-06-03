"""Small dataset utilities used by demo scripts."""

from __future__ import annotations

from typing import Tuple

import numpy as np


ArrayLike = np.ndarray


def train_test_split(X: ArrayLike, y: ArrayLike, test_size: float = 0.25, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    n = len(y)
    indices = rng.permutation(n)
    n_test = int(round(n * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class StandardScaler:
    """Minimal standard scaler, implemented to keep the project self-contained."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: ArrayLike):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted yet.")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        return self.fit(X).transform(X)


def make_binary_blobs(n_samples: int = 2000, random_state: int = 42) -> Tuple[ArrayLike, ArrayLike]:
    """Two-dimensional binary dataset for decision-boundary visualization."""
    rng = np.random.default_rng(random_state)
    n1 = n_samples // 2
    n2 = n_samples - n1
    cov = np.array([[1.2, 0.55], [0.55, 1.0]])
    X_pos = rng.multivariate_normal(mean=[1.7, 1.5], cov=cov, size=n1)
    X_neg = rng.multivariate_normal(mean=[-1.6, -1.3], cov=cov, size=n2)
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n1), -np.ones(n2)]).astype(int)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_two_moons(
    n_samples: int = 1200,
    noise: float = 0.12,
    random_state: int = 42,
) -> Tuple[ArrayLike, ArrayLike]:
    """Nonlinear two-moons binary dataset encoded with labels {-1, +1}."""
    rng = np.random.default_rng(random_state)
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    upper_theta = rng.uniform(0.0, np.pi, size=n_upper)
    lower_theta = rng.uniform(0.0, np.pi, size=n_lower)
    upper = np.column_stack([np.cos(upper_theta), np.sin(upper_theta)])
    lower = np.column_stack([1.0 - np.cos(lower_theta), 0.5 - np.sin(lower_theta)])

    X = np.vstack([upper, lower])
    X += rng.normal(scale=noise, size=X.shape)
    y = np.hstack([np.ones(n_upper), -np.ones(n_lower)]).astype(int)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def make_high_dimensional_binary(
    n_samples: int = 10000,
    n_features: int = 100,
    n_informative: int = 12,
    noise: float = 1.0,
    random_state: int = 42,
) -> Tuple[ArrayLike, ArrayLike]:
    """Synthetic high-dimensional linear classification task."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    true_w = np.zeros(n_features)
    true_w[:n_informative] = rng.normal(loc=0.0, scale=1.0, size=n_informative)
    logits = X @ true_w + noise * rng.normal(size=n_samples)
    y = np.where(logits >= 0.0, 1, -1).astype(int)
    return X, y


def make_multiclass_blobs(
    n_samples: int = 3000,
    n_features: int = 2,
    n_classes: int = 3,
    cluster_std: float = 0.9,
    random_state: int = 42,
) -> Tuple[ArrayLike, ArrayLike]:
    """Simple multiclass Gaussian blobs for one-vs-rest SVM demo."""
    rng = np.random.default_rng(random_state)
    angles = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
    centers = np.column_stack([3.0 * np.cos(angles), 3.0 * np.sin(angles)])
    if n_features > 2:
        centers = np.hstack([centers, np.zeros((n_classes, n_features - 2))])

    counts = np.full(n_classes, n_samples // n_classes)
    counts[: n_samples % n_classes] += 1
    X_parts = []
    y_parts = []
    for cls, count in enumerate(counts):
        X_parts.append(rng.normal(loc=centers[cls], scale=cluster_std, size=(count, n_features)))
        y_parts.append(np.full(count, cls, dtype=int))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]
