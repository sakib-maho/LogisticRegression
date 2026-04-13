"""CLI for training/evaluating logistic regression on CSV datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logistic_regression.dataset import load_binary_dataset
from logistic_regression.model import LogisticRegressionGD, accuracy_score


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="logreg-cli",
        description="Train and evaluate logistic regression on a CSV file.",
    )
    parser.add_argument("--data", type=Path, required=True, help="CSV path with 'label' column")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Gradient descent learning rate")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    features, labels, _ = load_binary_dataset(args.data)
    model = LogisticRegressionGD(learning_rate=args.learning_rate, epochs=args.epochs)
    model.fit(features, labels)
    predictions = [model.predict(row) for row in features]
    accuracy = accuracy_score(labels, predictions)
    print(json.dumps({"samples": len(labels), "accuracy": round(accuracy, 4)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
