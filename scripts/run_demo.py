"""Run a complete Pegasos SVM demo and save figures.

Usage:
    python scripts/run_demo.py

Generated figures are saved under output/figures/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    StandardScaler,
    make_binary_blobs,
    make_high_dimensional_binary,
    make_multiclass_blobs,
    train_test_split,
)
from src.metrics import confusion_matrix  # noqa: E402
from src.pegasos import BinaryPegasosSVM, OneVsRestPegasosSVM  # noqa: E402

FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_loss_curve():
    X, y = make_high_dimensional_binary(
        n_samples=20000, n_features=60, n_informative=10, noise=1.5, random_state=1
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = BinaryPegasosSVM(
        lambda_=1e-4,
        epochs=25,
        batch_size=128,
        random_state=1,
        record_history=True,
        objective_sample_size=5000,
    )
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    plt.figure(figsize=(7, 4.5))
    plt.plot(model.history_["epoch"], model.history_["objective"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Objective value")
    plt.title(f"Pegasos Linear SVM Training Loss, test acc={test_acc:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "loss_curve.png", dpi=180)
    plt.close()


def plot_decision_boundary():
    X, y = make_binary_blobs(n_samples=1800, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = BinaryPegasosSVM(
        lambda_=5e-4,
        epochs=40,
        batch_size=64,
        random_state=2,
        record_history=False,
    )
    model.fit(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(6.5, 5.5))
    plt.contourf(xx, yy, zz, levels=20, alpha=0.25)
    plt.contour(xx, yy, zz, levels=[-1, 0, 1], linestyles=["--", "-", "--"])
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=12, alpha=0.75)
    plt.xlabel("Standardized feature 1")
    plt.ylabel("Standardized feature 2")
    plt.title(f"Decision Boundary and Margins, test acc={test_acc:.3f}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "decision_boundary.png", dpi=180)
    plt.close()


def plot_scale_experiment():
    sizes = [1000, 3000, 10000, 30000, 80000]
    times = []
    accuracies = []

    # Generate the largest dataset once, then take prefixes for comparable experiments.
    X, y = make_high_dimensional_binary(
        n_samples=max(sizes), n_features=80, n_informative=12, noise=1.4, random_state=3
    )
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.25, random_state=3
    )

    for n in sizes:
        X_train = X_train_all[:n]
        y_train = y_train_all[:n]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_eval = scaler.transform(X_test)

        model = BinaryPegasosSVM(
            lambda_=1e-4,
            epochs=12,
            batch_size=256,
            random_state=3,
            record_history=False,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        acc = model.score(X_eval, y_test)
        times.append(elapsed)
        accuracies.append(acc)
        print(f"n={n:>6}, time={elapsed:>7.3f}s, test_acc={acc:.4f}")

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("Training samples")
    plt.ylabel("Training time (seconds)")
    plt.title("Training Time vs Sample Size")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_size_time.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(sizes, accuracies, marker="o")
    plt.xlabel("Training samples")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Sample Size")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sample_size_accuracy.png", dpi=180)
    plt.close()


def plot_multiclass_confusion_matrix():
    X, y = make_multiclass_blobs(n_samples=4500, n_features=2, n_classes=4, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = OneVsRestPegasosSVM(
        lambda_=5e-4,
        epochs=30,
        batch_size=64,
        random_state=4,
        record_history=False,
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = clf.score(X_test, y_test)
    cm, labels = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"OvR Pegasos SVM Confusion Matrix, acc={acc:.3f}")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "multiclass_confusion_matrix.png", dpi=180)
    plt.close()


def main():
    plot_loss_curve()
    plot_decision_boundary()
    plot_scale_experiment()
    plot_multiclass_confusion_matrix()
    print(f"Figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
