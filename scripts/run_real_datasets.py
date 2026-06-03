"""Run Pegasos SVM experiments on RCV1 and Amazon Review Polarity.

Examples:
    python scripts/run_real_datasets.py --dataset rcv1
    python scripts/run_real_datasets.py --dataset amazon --max-train-samples 20000
    python scripts/run_real_datasets.py --dataset both --epochs 5 --batch-size 512

Results are written to output/results/real_dataset_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pegasos import BinaryPegasosSVM  # noqa: E402


RESULT_DIR = ROOT / "output" / "results"
FIG_DIR = ROOT / "output" / "figures"
DEFAULT_OUTPUT = RESULT_DIR / "real_dataset_results.csv"
AMAZON_FULL_TRAIN_SAMPLES = 3_600_000
AMAZON_FULL_TEST_SAMPLES = 400_000


class DatasetLoadError(RuntimeError):
    """Raised when a dataset cannot be loaded or materialized."""


def _require_sklearn():
    try:
        from sklearn.datasets import fetch_rcv1
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(
            "Please install scikit-learn first: pip install -r requirements.txt"
        ) from exc
    return fetch_rcv1, HashingVectorizer, train_test_split


def _materialize_stream(split, limit: int, seed: int, shuffle_buffer: int):
    if hasattr(split, "shuffle"):
        try:
            split = split.shuffle(seed=seed, buffer_size=shuffle_buffer)
        except TypeError:
            split = split.shuffle(seed=seed)

    texts = []
    labels = []
    for row in itertools.islice(split, limit):
        title = str(row.get("title", "") or "")
        content = (
            row.get("content")
            or row.get("text")
            or row.get("review")
            or row.get("review_body")
            or ""
        )
        text = f"{title} {content}".strip()
        label = row.get("label")
        if label is None:
            raise ValueError("Amazon Polarity row does not contain a 'label' field.")
        if isinstance(label, str):
            y_value = 1 if label.lower() in {"1", "positive", "pos"} else -1
        else:
            y_value = 1 if int(label) > 0 else -1
        texts.append(text)
        labels.append(y_value)

    return texts, np.asarray(labels, dtype=int)


def _local_dataset_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise DatasetLoadError(
        f"Unsupported Amazon local file type: {path}. Use parquet, csv, json, or jsonl."
    )


def load_amazon_polarity(args):
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.hf_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.hf_timeout))

    _, HashingVectorizer, _ = _require_sklearn()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Please install the Hugging Face datasets package: pip install -r requirements.txt"
        ) from exc

    train_limit = AMAZON_FULL_TRAIN_SAMPLES if args.amazon_full else args.max_train_samples
    test_limit = AMAZON_FULL_TEST_SAMPLES if args.amazon_full else args.max_test_samples

    try:
        if args.amazon_train_file and args.amazon_test_file:
            if not args.amazon_train_file.exists():
                raise DatasetLoadError(f"Amazon train file does not exist: {args.amazon_train_file}")
            if not args.amazon_test_file.exists():
                raise DatasetLoadError(f"Amazon test file does not exist: {args.amazon_test_file}")
            train_kind = _local_dataset_kind(args.amazon_train_file)
            test_kind = _local_dataset_kind(args.amazon_test_file)
            train_split = load_dataset(train_kind, data_files=str(args.amazon_train_file), split="train")
            test_split = load_dataset(test_kind, data_files=str(args.amazon_test_file), split="train")
        elif args.amazon_train_file or args.amazon_test_file:
            raise DatasetLoadError("Pass both --amazon-train-file and --amazon-test-file.")
        else:
            train_split = load_dataset(args.amazon_dataset, split="train", streaming=True)
            test_split = load_dataset(args.amazon_dataset, split="test", streaming=True)
        train_texts, y_train = _materialize_stream(
            train_split, train_limit, args.random_state, args.amazon_shuffle_buffer
        )
        test_texts, y_test = _materialize_stream(
            test_split, test_limit, args.random_state + 1, args.amazon_shuffle_buffer
        )
    except DatasetLoadError:
        raise
    except Exception as exc:
        raise DatasetLoadError(
            "Amazon Review Polarity could not be downloaded from Hugging Face. "
            "This is usually a network timeout. Try again with --dataset amazon, "
            "reduce --max-train-samples/--max-test-samples, set HF_TOKEN, or set a mirror "
            "with HF_ENDPOINT. You can also pass --amazon-train-file and --amazon-test-file "
            "after downloading the parquet files manually."
        ) from exc

    vectorizer = HashingVectorizer(
        n_features=args.hash_features,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, args.ngram_max),
    )
    X_train = vectorizer.transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return {
        "name": "amazon_polarity",
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def load_rcv1(args):
    fetch_rcv1, _, train_test_split = _require_sklearn()
    try:
        data = fetch_rcv1(
            data_home=str(args.data_dir),
            subset="all",
            shuffle=True,
            random_state=args.random_state,
        )
    except Exception as exc:
        raise DatasetLoadError(
            "RCV1 could not be downloaded or loaded. Check the network connection and "
            "--data-dir cache path."
        ) from exc
    target_names = list(data.target_names)
    if args.rcv1_label not in target_names:
        available = ", ".join(target_names[:20])
        raise SystemExit(
            f"RCV1 label '{args.rcv1_label}' was not found. First available labels: {available}"
        )

    label_idx = target_names.index(args.rcv1_label)
    y_binary = np.asarray(data.target[:, label_idx].toarray()).ravel().astype(int)
    y = np.where(y_binary > 0, 1, -1)

    n_requested = args.max_train_samples + args.max_test_samples
    sample_idx = np.arange(data.data.shape[0])
    if n_requested < len(sample_idx):
        _, sample_idx = train_test_split(
            sample_idx,
            test_size=n_requested,
            random_state=args.random_state,
            stratify=y,
        )

    train_idx, test_idx = train_test_split(
        sample_idx,
        train_size=args.max_train_samples,
        test_size=args.max_test_samples,
        random_state=args.random_state,
        stratify=y[sample_idx],
    )
    return {
        "name": f"rcv1_{args.rcv1_label}",
        "X_train": data.data[train_idx],
        "X_test": data.data[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
    }


def run_experiment(dataset, args):
    verbose = args.verbose
    model = BinaryPegasosSVM(
        lambda_=args.lambda_,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        record_history=args.record_history,
        objective_sample_size=args.objective_sample_size,
        verbose=verbose,
        log_every=args.log_every,
    )
    start = time.perf_counter()
    model.fit(dataset["X_train"], dataset["y_train"])
    train_seconds = time.perf_counter() - start
    test_accuracy = model.score(dataset["X_test"], dataset["y_test"])

    y_train = dataset["y_train"]
    history = {key: list(values) for key, values in model.history_.items() if values}
    row = {
        "dataset": dataset["name"],
        "train_samples": dataset["X_train"].shape[0],
        "test_samples": dataset["X_test"].shape[0],
        "features": dataset["X_train"].shape[1],
        "positive_rate": float(np.mean(y_train == 1)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda": args.lambda_,
        "training_seconds": train_seconds,
        "test_accuracy": test_accuracy,
    }
    if history.get("objective"):
        row["final_objective"] = history["objective"][-1]
    else:
        row["final_objective"] = ""
    if history.get("accuracy"):
        row["train_sample_accuracy"] = history["accuracy"][-1]
    else:
        row["train_sample_accuracy"] = ""
    return row, history


def write_results(rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "train_samples",
        "test_samples",
        "features",
        "positive_rate",
        "epochs",
        "batch_size",
        "lambda",
        "training_seconds",
        "test_accuracy",
        "final_objective",
        "train_sample_accuracy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results(rows):
    print("\nResults")
    print("-" * 86)
    print(
        f"{'dataset':<22} {'train':>8} {'test':>8} {'features':>9} "
        f"{'pos_rate':>8} {'seconds':>9} {'accuracy':>9} {'objective':>10}"
    )
    for row in rows:
        objective = row["final_objective"]
        objective_text = "" if objective == "" else f"{objective:.4f}"
        print(
            f"{row['dataset']:<22} {row['train_samples']:>8} {row['test_samples']:>8} "
            f"{row['features']:>9} {row['positive_rate']:>8.3f} "
            f"{row['training_seconds']:>9.3f} {row['test_accuracy']:>9.4f} "
            f"{objective_text:>10}"
        )


def _safe_filename(text: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in clean.split("_") if part)


def _annotate_bars(ax, bars, fmt):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt(height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_report_figures(rows, histories, figure_dir: Path):
    if not rows:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped figure generation.", file=sys.stderr)
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"]
    names = [row["dataset"] for row in rows]
    x = np.arange(len(names))
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]
    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    fig.suptitle("Pegasos Linear SVM on Real Text Datasets", fontsize=17, fontweight="bold")

    acc_bars = axes[0, 0].bar(x, [row["test_accuracy"] for row in rows], color=bar_colors)
    axes[0, 0].set_title("Test Accuracy")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 0].set_ylabel("Accuracy")
    _annotate_bars(axes[0, 0], acc_bars, lambda value: f"{value:.3f}")

    time_bars = axes[0, 1].bar(x, [row["training_seconds"] for row in rows], color=bar_colors)
    axes[0, 1].set_title("Training Time")
    axes[0, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 1].set_ylabel("Seconds")
    _annotate_bars(axes[0, 1], time_bars, lambda value: f"{value:.1f}s")

    for idx, name in enumerate(names):
        history = histories.get(name, {})
        epochs = history.get("epoch", [])
        if epochs and history.get("objective"):
            axes[1, 0].plot(
                epochs,
                history["objective"],
                marker="o",
                linewidth=2,
                color=colors[idx % len(colors)],
                label=name,
            )
        if epochs and history.get("active_rate"):
            axes[1, 1].plot(
                epochs,
                history["active_rate"],
                marker="o",
                linewidth=2,
                color=colors[idx % len(colors)],
                label=name,
            )

    axes[1, 0].set_title("Objective Value by Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Objective")
    axes[1, 1].set_title("Margin-Violating Samples")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Active Rate")
    axes[1, 1].set_ylim(bottom=0)
    for ax in axes[1]:
        if ax.lines:
            ax.legend(frameon=True)
        else:
            ax.text(0.5, 0.5, "Enable history to draw curves", ha="center", va="center")

    fig.text(
        0.01,
        0.015,
        "Higher accuracy is better; lower objective and active rate usually indicate training progress.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    report_path = figure_dir / "real_dataset_report.png"
    fig.savefig(report_path, dpi=220)
    plt.close(fig)
    saved.append(report_path)

    if any(histories.values()):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
        for idx, name in enumerate(names):
            history = histories.get(name, {})
            epochs = history.get("epoch", [])
            if not epochs:
                continue
            color = colors[idx % len(colors)]
            if history.get("objective"):
                axes[0].plot(epochs, history["objective"], marker="o", color=color, label=name)
            if history.get("accuracy"):
                axes[1].plot(epochs, history["accuracy"], marker="o", color=color, label=name)
            if history.get("epoch_seconds"):
                axes[2].plot(epochs, history["epoch_seconds"], marker="o", color=color, label=name)
        axes[0].set_title("Objective")
        axes[1].set_title("Training Sample Accuracy")
        axes[2].set_title("Epoch Time")
        axes[0].set_ylabel("Value")
        axes[1].set_ylabel("Accuracy")
        axes[2].set_ylabel("Seconds")
        for ax in axes:
            ax.set_xlabel("Epoch")
            if ax.lines:
                ax.legend(frameon=True)
        fig.suptitle("Training Diagnostics", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        curves_path = figure_dir / "real_dataset_training_curves.png"
        fig.savefig(curves_path, dpi=220)
        plt.close(fig)
        saved.append(curves_path)

    for name in names:
        history = histories.get(name, {})
        epochs = history.get("epoch", [])
        if not epochs or not history.get("objective"):
            continue
        fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
        color = "#2a9d8f"
        ax1.plot(epochs, history["objective"], marker="o", linewidth=2.2, color=color)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Objective", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax2 = ax1.twinx()
        color = "#e76f51"
        ax2.plot(epochs, history.get("accuracy", []), marker="s", linewidth=2.2, color=color)
        ax2.set_ylabel("Training Sample Accuracy", color=color)
        ax2.tick_params(axis="y", labelcolor=color)
        ax1.set_title(f"{name} Training Curve", fontsize=14, fontweight="bold")
        fig.tight_layout()
        detail_path = figure_dir / f"{_safe_filename(name)}_training_curve.png"
        fig.savefig(detail_path, dpi=220)
        plt.close(fig)
        saved.append(detail_path)

    return saved


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["rcv1", "amazon", "both"], default="both")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-test-samples", type=int, default=10000)
    parser.add_argument("--record-history", dest="record_history", action="store_true", default=True)
    parser.add_argument("--no-record-history", dest="record_history", action="store_false")
    parser.add_argument("--objective-sample-size", type=int, default=5000)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--rcv1-label", default="CCAT")
    parser.add_argument("--amazon-dataset", default="fancyzhx/amazon_polarity")
    parser.add_argument("--amazon-full", action="store_true")
    parser.add_argument("--amazon-train-file", type=Path)
    parser.add_argument("--amazon-test-file", type=Path)
    parser.add_argument("--amazon-shuffle-buffer", type=int, default=10000)
    parser.add_argument("--hf-timeout", type=int, default=120)
    parser.add_argument("--hash-features", type=int, default=2**18)
    parser.add_argument("--ngram-max", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    loaders = []
    if args.dataset in {"rcv1", "both"}:
        loaders.append(load_rcv1)
    if args.dataset in {"amazon", "both"}:
        loaders.append(load_amazon_polarity)

    rows = []
    histories = {}
    for load_dataset in loaders:
        try:
            dataset = load_dataset(args)
            print(
                f"Loaded {dataset['name']}: "
                f"{dataset['X_train'].shape[0]} train, {dataset['X_test'].shape[0]} test, "
                f"{dataset['X_train'].shape[1]} features"
            )
            print(f"Training {dataset['name']} ...")
            row, history = run_experiment(dataset, args)
            rows.append(row)
            histories[dataset["name"]] = history
            write_results(rows, args.output)
            print(f"Saved partial results to: {args.output}")
            if not args.no_figures:
                figure_paths = save_report_figures(rows, histories, args.figure_dir)
                if figure_paths:
                    print("Saved figures:")
                    for path in figure_paths:
                        print(f"  {path}")
        except DatasetLoadError as exc:
            print(f"\nSkipped dataset: {exc}", file=sys.stderr)
            if args.dataset != "both" or not rows:
                raise SystemExit(1) from exc

    write_results(rows, args.output)
    figure_paths = [] if args.no_figures else save_report_figures(rows, histories, args.figure_dir)
    print_results(rows)
    print(f"\nSaved CSV to: {args.output}")
    if figure_paths:
        print("Saved final figures:")
        for path in figure_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
