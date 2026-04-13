"""Minimal logistic regression implementation with gradient descent."""

from __future__ import annotations

import math


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 300) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        if not features:
            raise ValueError("features must not be empty")
        feature_count = len(features[0])
        self.weights = [0.0 for _ in range(feature_count)]
        self.bias = 0.0

        sample_count = len(features)
        for _ in range(self.epochs):
            dw = [0.0 for _ in range(feature_count)]
            db = 0.0
            for row, label in zip(features, labels):
                linear = sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
                prediction = _sigmoid(linear)
                error = prediction - label
                for idx in range(feature_count):
                    dw[idx] += error * row[idx]
                db += error

            for idx in range(feature_count):
                self.weights[idx] -= self.learning_rate * (dw[idx] / sample_count)
            self.bias -= self.learning_rate * (db / sample_count)

    def predict_proba(self, row: list[float]) -> float:
        linear = sum(weight * value for weight, value in zip(self.weights, row)) + self.bias
        return _sigmoid(linear)

    def predict(self, row: list[float]) -> int:
        return 1 if self.predict_proba(row) >= 0.5 else 0


def accuracy_score(labels: list[int], predictions: list[int]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    return correct / len(labels)
