"""Run a nonlinear SVM experiment with an RBF kernel trained by SMO.

Generated artifacts:
    output/figures/nonlinear_svm_decision_boundary.png
    output/figures/nonlinear_svm_training_diagnostics.png
    output/results/nonlinear_svm_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import StandardScaler, make_two_moons, train_test_split  # noqa: E402
from src.kernel_svm import BinaryKernelSVM  # noqa: E402
from src.pegasos import BinaryPegasosSVM  # noqa: E402


FIG_DIR = ROOT / "output" / "figures"
RESULT_DIR = ROOT / "output" / "results"


def _decision_grid(model, X, grid_size):
    x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
    y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = model.decision_function(grid).reshape(xx.shape)
    return xx, yy, zz


def _plot_model_boundary(ax, model, X_train, y_train, X_test, y_test, title, grid_size, support_vectors=None):
    xx, yy, zz = _decision_grid(model, np.vstack([X_train, X_test]), grid_size)
    ax.contourf(xx, yy, zz, levels=24, cmap="RdBu_r", alpha=0.28)
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["#555555", "#111111", "#555555"], linestyles=["--", "-", "--"])
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap="RdBu_r",
        s=18,
        edgecolors="none",
        alpha=0.82,
        label="train",
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="RdBu_r",
        s=34,
        marker="x",
        linewidths=1.2,
        alpha=0.95,
        label="test",
    )
    if support_vectors is not None and len(support_vectors) > 0:
        ax.scatter(
            support_vectors[:, 0],
            support_vectors[:, 1],
            s=62,
            facecolors="none",
            edgecolors="#222222",
            linewidths=0.9,
            label="support vectors",
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.legend(loc="upper right", frameon=True, fontsize=8)


def _write_results(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "train_samples",
                "test_samples",
                "training_seconds",
                "test_accuracy",
                "support_vectors",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = make_two_moons(n_samples=args.samples, noise=args.noise, random_state=args.random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    linear = BinaryPegasosSVM(
        lambda_=args.lambda_,
        epochs=args.linear_epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        record_history=True,
        verbose=args.verbose,
    )
    start = time.perf_counter()
    linear.fit(X_train, y_train)
    linear_seconds = time.perf_counter() - start
    linear_acc = linear.score(X_test, y_test)

    kernel = BinaryKernelSVM(
        C=args.C,
        kernel="rbf",
        gamma=args.gamma,
        tol=args.tol,
        max_passes=args.max_passes,
        max_iter=args.max_iter,
        random_state=args.random_state,
        verbose=args.verbose,
    )
    start = time.perf_counter()
    kernel.fit(X_train, y_train)
    kernel_seconds = time.perf_counter() - start
    kernel_acc = kernel.score(X_test, y_test)

    rows = [
        {
            "model": "linear_pegasos",
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "training_seconds": linear_seconds,
            "test_accuracy": linear_acc,
            "support_vectors": "",
        },
        {
            "model": "rbf_kernel_smo",
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "training_seconds": kernel_seconds,
            "test_accuracy": kernel_acc,
            "support_vectors": len(kernel.support_),
        },
    ]
    results_path = RESULT_DIR / "nonlinear_svm_results.csv"
    _write_results(rows, results_path)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True, sharey=True)
    _plot_model_boundary(
        axes[0],
        linear,
        X_train,
        y_train,
        X_test,
        y_test,
        f"Linear Pegasos SVM, acc={linear_acc:.3f}",
        args.grid_size,
    )
    _plot_model_boundary(
        axes[1],
        kernel,
        X_train,
        y_train,
        X_test,
        y_test,
        f"RBF Kernel SVM via SMO, acc={kernel_acc:.3f}",
        args.grid_size,
        support_vectors=kernel.support_vectors_,
    )
    fig.suptitle("Linear vs Nonlinear SVM on Two-Moons Data", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    boundary_path = FIG_DIR / "nonlinear_svm_decision_boundary.png"
    fig.savefig(boundary_path, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(linear.history_["epoch"], linear.history_["objective"], marker="o", color="#2a9d8f")
    axes[0].set_title("Pegasos Objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Objective")

    axes[1].plot(kernel.history_["iteration"], kernel.history_["training_accuracy"], marker="o", color="#e76f51")
    axes[1].set_title("SMO Training Accuracy")
    axes[1].set_xlabel("Outer Iteration")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)

    axes[2].plot(kernel.history_["iteration"], kernel.history_["support_vectors"], marker="o", color="#457b9d")
    axes[2].set_title("Support Vector Count")
    axes[2].set_xlabel("Outer Iteration")
    axes[2].set_ylabel("Count")

    fig.suptitle("Nonlinear SVM Training Diagnostics", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    diagnostics_path = FIG_DIR / "nonlinear_svm_training_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=220)
    plt.close(fig)

    print("Results")
    print("-" * 74)
    for row in rows:
        print(
            f"{row['model']:<18} "
            f"time={row['training_seconds']:.3f}s "
            f"acc={row['test_accuracy']:.4f} "
            f"support_vectors={row['support_vectors']}"
        )
    print(f"Saved CSV to: {results_path}")
    print(f"Saved figures to: {boundary_path}")
    print(f"Saved figures to: {diagnostics_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--noise", type=float, default=0.16)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--linear-epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-3)
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
