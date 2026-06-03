"""Template for running Pegasos SVM on LIBSVM-format datasets.

This script is optional. It does not download datasets automatically.
After installing scikit-learn, place a dataset such as rcv1_train.binary in data/
and run:

    python scripts/run_real_dataset_template.py --path data/rcv1_train.binary
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pegasos import BinaryPegasosSVM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to a LIBSVM-format binary classification dataset.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        from sklearn.datasets import load_svmlight_file
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        raise SystemExit(
            "Please install scikit-learn to use this template: pip install -r requirements.txt"
        ) from exc

    X, y = load_svmlight_file(args.path)
    y = (y > 0).astype(int) * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    X_train = normalize(X_train, norm="l2", copy=False)
    X_test = normalize(X_test, norm="l2", copy=False)

    model = BinaryPegasosSVM(lambda_=args.lambda_, epochs=args.epochs, batch_size=args.batch_size)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    print(f"Training time: {elapsed:.3f}s")
    print(f"Test accuracy: {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
