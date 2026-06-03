"""Kernel SVM trained with a simplified SMO algorithm.

This implementation is intended for small nonlinear datasets and teaching
experiments. For large sparse text datasets, use BinaryPegasosSVM instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass
class BinaryKernelSVM:
    """Binary soft-margin kernel SVM optimized with simplified SMO."""

    C: float = 1.0
    kernel: str = "rbf"
    gamma: Optional[float] = None
    degree: int = 3
    coef0: float = 1.0
    tol: float = 1e-3
    max_passes: int = 8
    max_iter: int = 1000
    random_state: Optional[int] = 42
    verbose: int = 0

    X_: Optional[ArrayLike] = field(default=None, init=False)
    y_: Optional[ArrayLike] = field(default=None, init=False)
    alphas_: Optional[ArrayLike] = field(default=None, init=False)
    b_: float = field(default=0.0, init=False)
    support_: Optional[ArrayLike] = field(default=None, init=False)
    support_vectors_: Optional[ArrayLike] = field(default=None, init=False)
    support_labels_: Optional[ArrayLike] = field(default=None, init=False)
    support_alphas_: Optional[ArrayLike] = field(default=None, init=False)
    history_: Dict[str, List[float]] = field(default_factory=dict, init=False)

    def _check_y(self, y: ArrayLike) -> ArrayLike:
        y = np.asarray(y).ravel().astype(float)
        unique = set(np.unique(y).tolist())
        if unique <= {-1.0, 1.0}:
            return y
        if unique <= {0.0, 1.0}:
            return np.where(y > 0, 1.0, -1.0)
        raise ValueError("BinaryKernelSVM requires labels encoded as {-1,+1} or {0,1}.")

    def _gamma(self, n_features: int) -> float:
        return 1.0 / n_features if self.gamma is None else self.gamma

    def _kernel_matrix(self, X: ArrayLike, Z: ArrayLike) -> ArrayLike:
        if self.kernel == "linear":
            return X @ Z.T
        if self.kernel == "poly":
            return (self._gamma(X.shape[1]) * (X @ Z.T) + self.coef0) ** self.degree
        if self.kernel == "rbf":
            x_norm = np.sum(X * X, axis=1)[:, None]
            z_norm = np.sum(Z * Z, axis=1)[None, :]
            sq_dist = np.maximum(x_norm + z_norm - 2.0 * (X @ Z.T), 0.0)
            return np.exp(-self._gamma(X.shape[1]) * sq_dist)
        raise ValueError("kernel must be one of {'linear', 'poly', 'rbf'}.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BinaryKernelSVM":
        X = np.asarray(X, dtype=float)
        y = self._check_y(y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if len(y) != X.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples.")
        if self.C <= 0:
            raise ValueError("C must be positive.")

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        K = self._kernel_matrix(X, X)
        alphas = np.zeros(n_samples, dtype=float)
        b = 0.0
        passes = 0
        iteration = 0
        start_time = time.perf_counter()
        self.history_ = {
            "iteration": [],
            "changed_alphas": [],
            "support_vectors": [],
            "training_accuracy": [],
            "elapsed_seconds": [],
        }

        while passes < self.max_passes and iteration < self.max_iter:
            changed = 0

            for i in range(n_samples):
                Ei = float((alphas * y) @ K[:, i] + b - y[i])
                violates_lower = y[i] * Ei < -self.tol and alphas[i] < self.C
                violates_upper = y[i] * Ei > self.tol and alphas[i] > 0.0
                if not (violates_lower or violates_upper):
                    continue

                j = int(rng.integers(0, n_samples - 1))
                if j >= i:
                    j += 1

                Ej = float((alphas * y) @ K[:, j] + b - y[j])
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] != y[j]:
                    lower = max(0.0, alpha_j_old - alpha_i_old)
                    upper = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    lower = max(0.0, alpha_i_old + alpha_j_old - self.C)
                    upper = min(self.C, alpha_i_old + alpha_j_old)
                if lower == upper:
                    continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0.0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], lower, upper)
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = (
                    b
                    - Ei
                    - y[i] * (alphas[i] - alpha_i_old) * K[i, i]
                    - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                )
                b2 = (
                    b
                    - Ej
                    - y[i] * (alphas[i] - alpha_i_old) * K[i, j]
                    - y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                )
                if 0.0 < alphas[i] < self.C:
                    b = b1
                elif 0.0 < alphas[j] < self.C:
                    b = b2
                else:
                    b = 0.5 * (b1 + b2)

                changed += 1

            iteration += 1
            passes = passes + 1 if changed == 0 else 0
            scores = (alphas * y) @ K + b
            acc = float(np.mean(np.where(scores >= 0.0, 1.0, -1.0) == y))
            support_count = int(np.count_nonzero(alphas > 1e-6))
            self.history_["iteration"].append(float(iteration))
            self.history_["changed_alphas"].append(float(changed))
            self.history_["support_vectors"].append(float(support_count))
            self.history_["training_accuracy"].append(acc)
            self.history_["elapsed_seconds"].append(float(time.perf_counter() - start_time))

            if self.verbose:
                print(
                    "[SMO] "
                    f"iter={iteration} "
                    f"changed={changed} "
                    f"passes={passes}/{self.max_passes} "
                    f"support={support_count} "
                    f"train_acc={acc:.4f}"
                )

        support = alphas > 1e-6
        self.X_ = X
        self.y_ = y
        self.alphas_ = alphas
        self.b_ = float(b)
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.support_labels_ = y[support]
        self.support_alphas_ = alphas[support]
        return self

    def decision_function(self, X: ArrayLike) -> ArrayLike:
        if self.support_vectors_ is None or self.support_alphas_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = np.asarray(X, dtype=float)
        K = self._kernel_matrix(X, self.support_vectors_)
        return K @ (self.support_alphas_ * self.support_labels_) + self.b_

    def predict(self, X: ArrayLike) -> ArrayLike:
        return np.where(self.decision_function(X) >= 0.0, 1, -1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = self._check_y(y)
        return float(np.mean(self.predict(X) == y_true))
