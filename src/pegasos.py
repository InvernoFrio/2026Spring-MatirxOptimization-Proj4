"""Pegasos linear SVM implementation.

This module intentionally avoids calling sklearn models.  It implements:
- BinaryPegasosSVM: binary linear SVM trained with mini-batch Pegasos.
- OneVsRestPegasosSVM: multiclass wrapper using one-vs-rest binary models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy import sparse
except ImportError:  # pragma: no cover - sparse support is optional.
    sparse = None


ArrayLike = np.ndarray


def _is_sparse_matrix(X) -> bool:
    return sparse is not None and sparse.issparse(X)


def _as_float_features(X):
    if _is_sparse_matrix(X):
        return X.tocsr().astype(float)
    return np.asarray(X, dtype=float)


def _signed_feature_mean(X, y: ArrayLike) -> ArrayLike:
    """Mean of y_i * x_i for dense or sparse feature batches."""
    return np.asarray(X.T @ y, dtype=float).reshape(-1) / X.shape[0]


@dataclass
class BinaryPegasosSVM:
    """Binary linear SVM optimized by mini-batch Pegasos.

    The optimized objective is approximately

        lambda_/2 * ||w||^2 + mean(max(0, 1 - y * (Xw + b)))

    with y encoded as {-1, +1}.

    Parameters
    ----------
    lambda_:
        L2 regularization strength. Larger lambda_ means stronger regularization.
    epochs:
        Number of passes over the training set.
    batch_size:
        Mini-batch size. batch_size=1 is the original stochastic Pegasos style.
    fit_intercept:
        Whether to learn a bias term b. The bias term is not L2-regularized.
    project:
        Whether to project w to the ball ||w|| <= 1/sqrt(lambda_). This is used in
        the original Pegasos convergence analysis.
    random_state:
        Random seed for shuffling batches.
    record_history:
        Whether to record objective values after each epoch.
    objective_sample_size:
        If set, objective is estimated on a fixed subset to keep large runs fast.
    verbose:
        0 is silent, 1 prints epoch summaries, 2 also prints periodic batch details.
    log_every:
        Batch interval for verbose=2. If 0, one message is printed every 10 batches.
    """

    lambda_: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    fit_intercept: bool = True
    project: bool = True
    random_state: Optional[int] = 42
    record_history: bool = True
    objective_sample_size: Optional[int] = 5000
    verbose: int = 0
    log_every: int = 0

    w_: Optional[ArrayLike] = field(default=None, init=False)
    b_: float = field(default=0.0, init=False)
    history_: Dict[str, List[float]] = field(default_factory=dict, init=False)

    def _check_y(self, y: ArrayLike) -> ArrayLike:
        y = np.asarray(y).ravel().astype(float)
        unique = set(np.unique(y).tolist())
        if unique <= {-1.0, 1.0}:
            return y
        if unique <= {0.0, 1.0}:
            return np.where(y > 0, 1.0, -1.0)
        raise ValueError("BinaryPegasosSVM requires labels encoded as {-1,+1} or {0,1}.")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "BinaryPegasosSVM":
        X = _as_float_features(X)
        y = self._check_y(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if len(y) != X.shape[0]:
            raise ValueError("X and y have inconsistent numbers of samples.")
        if self.lambda_ <= 0:
            raise ValueError("lambda_ must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.w_ = np.zeros(n_features, dtype=float)
        self.b_ = 0.0
        self.history_ = {
            "epoch": [],
            "objective": [],
            "accuracy": [],
            "active_rate": [],
            "epoch_seconds": [],
        }

        if self.record_history:
            if self.objective_sample_size is not None and n_samples > self.objective_sample_size:
                hist_idx = rng.choice(n_samples, size=self.objective_sample_size, replace=False)
            else:
                hist_idx = np.arange(n_samples)
        else:
            hist_idx = np.array([], dtype=int)

        n_batches = int(np.ceil(n_samples / self.batch_size))
        log_every = self.log_every if self.log_every > 0 else 10
        t = 0
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            epoch_active = 0
            epoch_seen = 0
            indices = rng.permutation(n_samples)
            for batch_no, start in enumerate(range(0, n_samples, self.batch_size), start=1):
                t += 1
                batch_idx = indices[start : start + self.batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx]
                eta = 1.0 / (self.lambda_ * t)

                margins = yb * (Xb @ self.w_ + self.b_)
                active = margins < 1.0
                active_count = int(np.count_nonzero(active))
                epoch_active += active_count
                epoch_seen += len(yb)

                # Regularization shrinkage.
                self.w_ *= (1.0 - eta * self.lambda_)

                # Hinge-loss update for margin-violating examples.
                if np.any(active):
                    correction = _signed_feature_mean(Xb[active], yb[active])
                    self.w_ += eta * correction
                    if self.fit_intercept:
                        self.b_ += eta * yb[active].mean()

                if self.project:
                    norm = np.linalg.norm(self.w_)
                    radius = 1.0 / np.sqrt(self.lambda_)
                    if norm > radius:
                        self.w_ *= radius / norm

                if self.verbose >= 2 and (batch_no == 1 or batch_no % log_every == 0 or batch_no == n_batches):
                    active_rate = active_count / len(yb)
                    print(
                        "[Pegasos] "
                        f"epoch={epoch}/{self.epochs} "
                        f"batch={batch_no}/{n_batches} "
                        f"eta={eta:.3e} "
                        f"batch_active_rate={active_rate:.3f}"
                    )

            epoch_seconds = time.perf_counter() - epoch_start
            active_rate = epoch_active / max(epoch_seen, 1)
            if self.record_history:
                obj = self.objective(X[hist_idx], y[hist_idx])
                acc = self.score(X[hist_idx], y[hist_idx])
                self.history_["epoch"].append(float(epoch))
                self.history_["objective"].append(float(obj))
                self.history_["accuracy"].append(float(acc))
                self.history_["active_rate"].append(float(active_rate))
                self.history_["epoch_seconds"].append(float(epoch_seconds))
                if self.verbose:
                    print(
                        "[Pegasos] "
                        f"epoch={epoch}/{self.epochs} "
                        f"seconds={epoch_seconds:.3f} "
                        f"active_rate={active_rate:.3f} "
                        f"objective={obj:.6f} "
                        f"sample_acc={acc:.4f} "
                        f"||w||={np.linalg.norm(self.w_):.3f}"
                    )
            elif self.verbose:
                print(
                    "[Pegasos] "
                    f"epoch={epoch}/{self.epochs} "
                    f"seconds={epoch_seconds:.3f} "
                    f"active_rate={active_rate:.3f} "
                    f"||w||={np.linalg.norm(self.w_):.3f}"
                )

        return self

    def decision_function(self, X: ArrayLike) -> ArrayLike:
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = _as_float_features(X)
        return X @ self.w_ + self.b_

    def predict(self, X: ArrayLike) -> ArrayLike:
        scores = self.decision_function(X)
        return np.where(scores >= 0.0, 1, -1)

    def objective(self, X: ArrayLike, y: ArrayLike) -> float:
        if self.w_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = _as_float_features(X)
        y = self._check_y(y)
        margins = y * self.decision_function(X)
        hinge = np.maximum(0.0, 1.0 - margins)
        return 0.5 * self.lambda_ * float(np.dot(self.w_, self.w_)) + float(np.mean(hinge))

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y_true = self._check_y(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y_true))


@dataclass
class OneVsRestPegasosSVM:
    """Multiclass SVM wrapper built from multiple BinaryPegasosSVM models."""

    lambda_: float = 1e-4
    epochs: int = 20
    batch_size: int = 64
    fit_intercept: bool = True
    project: bool = True
    random_state: Optional[int] = 42
    record_history: bool = False
    objective_sample_size: Optional[int] = 5000
    verbose: int = 0
    log_every: int = 0

    classes_: Optional[ArrayLike] = field(default=None, init=False)
    models_: List[BinaryPegasosSVM] = field(default_factory=list, init=False)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "OneVsRestPegasosSVM":
        X = _as_float_features(X)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.models_ = []

        for idx, cls in enumerate(self.classes_):
            y_bin = np.where(y == cls, 1, -1)
            seed = None if self.random_state is None else self.random_state + idx
            model = BinaryPegasosSVM(
                lambda_=self.lambda_,
                epochs=self.epochs,
                batch_size=self.batch_size,
                fit_intercept=self.fit_intercept,
                project=self.project,
                random_state=seed,
                record_history=self.record_history,
                objective_sample_size=self.objective_sample_size,
                verbose=self.verbose,
                log_every=self.log_every,
            )
            model.fit(X, y_bin)
            self.models_.append(model)
        return self

    def decision_function(self, X: ArrayLike) -> ArrayLike:
        if self.classes_ is None or not self.models_:
            raise RuntimeError("Model is not fitted yet.")
        scores = [model.decision_function(X) for model in self.models_]
        return np.vstack(scores).T

    def predict(self, X: ArrayLike) -> ArrayLike:
        scores = self.decision_function(X)
        best = np.argmax(scores, axis=1)
        return self.classes_[best]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))
