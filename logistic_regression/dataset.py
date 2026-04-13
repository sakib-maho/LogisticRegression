"""Dataset helpers for CSV-based binary classification."""

from __future__ import annotations

import csv
from pathlib import Path


def load_binary_dataset(path: Path) -> tuple[list[list[float]], list[int], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError("CSV must include a 'label' column")
        feature_names = [name for name in reader.fieldnames if name != "label"]
        features: list[list[float]] = []
        labels: list[int] = []
        for row in reader:
            features.append([float(row[name]) for name in feature_names])
            labels.append(int(row["label"]))
    return features, labels, feature_names
