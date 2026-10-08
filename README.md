# Challenge ML

A working project for the `Toxic gas identification` challenge.

## Project Structure

### Data

- `data/raw/` contains the challenge input files:
  - `x_train.csv`
  - `y_train.csv`
  - `x_test.csv`
- `data/processed/` contains pipeline outputs, such as submission CSVs and intermediate files.

The raw data files are not included in this repository to keep it small. The scripts expect `data/raw/x_train.csv`, `data/raw/y_train.csv`, and `data/raw/x_test.csv`. The pipeline cannot run without these files.

### Code

- [src/challenge_ml/feature.py](src/challenge_ml/feature.py) prepares the data. Its main functions are:
  - `load_datasets()`
  - `feature_engineering(features)`
  - `build_training_data(use_features=True)`

  It loads the CSV files, prepares raw or engineered features, and builds `X_train`, `y_train`, `X_test`, and `test_ids`.

- [src/challenge_ml/split.py](src/challenge_ml/split.py) provides local validation splits. Its main functions are:
  - `random_split(...)`
  - `humidity_split(...)`
  - `random_split_from_features(...)`
  - `humidity_split_from_features(...)`

  These functions create random splits or splits based on `Humidity`.

- [src/challenge_ml/train_xgboost.py](src/challenge_ml/train_xgboost.py) contains the main training, evaluation, and submission pipeline. Its main functions are:
  - `build_xgb_model(...)`
  - `evaluate_model(...)`
  - `compare_raw_vs_features()`
  - `run_regularized_feature_model()`
  - `train_full_and_predict(...)`
  - `build_raw_submission()`
  - `build_regularized_feature_submission()`

## Pipeline Overview

The project supports two workflows.

### 1. Local Evaluation

Use the available splits to compare approaches before creating a submission:

- `random`
- `humidity`

These splits help compare raw and engineered features, try a more regularized model, and assess local robustness.

### 2. Final Submission

After choosing a model, the pipeline trains on all of `x_train` and `y_train`, predicts on `x_test`, and writes a CSV file to `data/processed/`. This is handled by `train_full_and_predict(...)`.

## Commands

### Set Up the Environment

```bash
uv sync
uv pip install -e .
```

These commands install the dependencies and the local `challenge_ml` package in editable mode, so imports such as `from challenge_ml...` work.

### Run Local Experiments

From the repository root, run:

```bash
uv run python -m challenge_ml.train_xgboost
```

This runs the raw-versus-features comparison, the regularized feature model, and a simple local ensemble.

### Generate a Raw-Model Submission

```bash
uv run python -c "from challenge_ml.train_xgboost import build_raw_submission; build_raw_submission()"
```

The generated file is `data/processed/submission_xgboost_raw.csv`.

### Generate a Regularized Feature-Model Submission

```bash
uv run python -c "from challenge_ml.train_xgboost import build_regularized_feature_submission; build_regularized_feature_submission()"
```

The generated file is `data/processed/submission_xgboost_features_regularized.csv`.

### Install Without `uv`

Alternatively, create a virtual environment and install the requirements with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Useful Imports

### Load Prepared Data

```python
from challenge_ml.feature import build_training_data

X_train, y_train, X_test, test_ids = build_training_data(use_features=True)
```

### Create a Local Split

```python
from challenge_ml.split import random_split_from_features, humidity_split_from_features

X_train, X_valid, y_train, y_valid = random_split_from_features(use_features=True)
```

Or use the humidity-based split with raw features:

```python
X_train, X_valid, y_train, y_valid = humidity_split_from_features(use_features=False)
```

### Generate a Submission from Python

```python
from challenge_ml.train_xgboost import build_raw_submission

model, submission = build_raw_submission()
```

## Submission Format

The final CSV must contain `ID` and columns `c01` through `c23`. The pipeline sets `c15` to `0.0` automatically.
