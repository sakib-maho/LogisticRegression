from pathlib import Path
from subprocess import run
import json
import unittest

from logistic_regression.dataset import load_binary_dataset
from logistic_regression.model import LogisticRegressionGD, accuracy_score


class LogisticRegressionTests(unittest.TestCase):
    def test_training_reaches_reasonable_accuracy(self) -> None:
        features, labels, _ = load_binary_dataset(Path("data/sample_binary.csv"))
        model = LogisticRegressionGD(learning_rate=0.05, epochs=800)
        model.fit(features, labels)
        predictions = [model.predict(row) for row in features]
        self.assertGreaterEqual(accuracy_score(labels, predictions), 0.75)

    def test_cli_output(self) -> None:
        result = run(
            ["python3", "cli.py", "--data", "data/sample_binary.csv", "--epochs", "400"],
            check=True,
            text=True,
            capture_output=True,
        )
        payload = json.loads(result.stdout)
        self.assertIn("accuracy", payload)
        self.assertEqual(payload["samples"], 8)


if __name__ == "__main__":
    unittest.main()
