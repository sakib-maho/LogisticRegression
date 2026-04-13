# Logistic Regression Showcase

<!-- BrandCloud:readme-standard -->
[![Maintained](https://img.shields.io/badge/Maintained-yes-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Showcase](https://img.shields.io/badge/Portfolio-Showcase-blue.svg)](#)

_Part of the `sakib-maho` project showcase series with consistent documentation and quality standards._

This repository has been upgraded into a reproducible logistic regression mini-project.
The notebook remains available, while a clean Python package and CLI provide scriptable training workflows.

## Features

- Lightweight logistic regression implementation (gradient descent)
- CSV dataset loader with `label` target column
- CLI for training and evaluating model accuracy
- Unit tests for model behavior and CLI output
- Sample dataset for quick experimentation

## Quick Start

```bash
python3 cli.py --data data/sample_binary.csv --epochs 400 --learning-rate 0.1
```

## Run Tests

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## Project Structure

```text
LogisticRegression/
├── LogisticRegression.ipynb
├── cli.py
├── logistic_regression/
│   ├── dataset.py
│   └── model.py
├── data/sample_binary.csv
└── tests/test_model.py
```

## License

MIT License. See `LICENSE`.
