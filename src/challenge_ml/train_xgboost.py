from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from challenge_ml.feature import build_training_data
from challenge_ml.split import humidity_split_from_features, random_split_from_features


SUBMISSION_COLUMNS = [
    "ID",
    "c01",
    "c02",
    "c03",
    "c04",
    "c05",
    "c06",
    "c07",
    "c08",
    "c09",
    "c10",
    "c11",
    "c12",
    "c13",
    "c14",
    "c15",
    "c16",
    "c17",
    "c18",
    "c19",
    "c20",
    "c21",
    "c22",
    "c23",
]

BASELINE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

REGULARIZED_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
}


def build_xgb_model(**xgb_params):
    """Build a multi-output XGBoost regressor."""
    default_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    }
    default_params.update(xgb_params)
    return MultiOutputRegressor(XGBRegressor(**default_params))


def evaluate_model(y_true, y_pred):
    """Compute validation metrics after clipping predictions to [0, 1]."""
    y_pred = np.clip(y_pred, 0, 1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def print_metrics(name, metrics):
    print(name)
    print("MAE :", round(metrics["mae"], 5))
    print("RMSE:", round(metrics["rmse"], 5))
    print("R2  :", round(metrics["r2"], 5))


def train_and_evaluate(X_train, X_valid, y_train, y_valid, *, model_name: str, xgb_params):
    """Train a model on a split, predict on validation, and compute metrics."""
    model = build_xgb_model(**xgb_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    metrics = evaluate_model(y_valid, y_pred)
    print_metrics(model_name, metrics)

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": np.clip(y_pred, 0, 1),
        "y_true": y_valid,
    }


def get_split(split_name: str, *, use_features: bool):
    if split_name == "random":
        return random_split_from_features(use_features=use_features)
    if split_name == "humidity":
        return humidity_split_from_features(use_features=use_features)
    raise ValueError(f"Unknown split: {split_name}")


def run_experiment(*, split_name: str, dataset_name: str, xgb_params):
    use_features = dataset_name == "features"
    X_train, X_valid, y_train, y_valid = get_split(
        split_name,
        use_features=use_features,
    )
    return train_and_evaluate(
        X_train,
        X_valid,
        y_train,
        y_valid,
        model_name=f"XGBoost - {dataset_name} - {split_name}",
        xgb_params=xgb_params,
    )


def compare_raw_vs_features():
    """Compare raw and engineered features on both validation splits."""
    results = {}
    for dataset_name in ("raw", "features"):
        for split_name in ("random", "humidity"):
            key = f"{dataset_name}_{split_name}"
            results[key] = run_experiment(
                split_name=split_name,
                dataset_name=dataset_name,
                xgb_params=BASELINE_PARAMS,
            )
            print()
    return results


def run_regularized_feature_model():
    """Evaluate a more regularized feature-based model."""
    results = {}
    for split_name in ("random", "humidity"):
        key = f"regularized_features_{split_name}"
        results[key] = run_experiment(
            split_name=split_name,
            dataset_name="features",
            xgb_params=REGULARIZED_PARAMS,
        )
        print()
    return results


def evaluate_ensemble(run_a, run_b, *, name: str):
    """Average two validation prediction sets and evaluate the ensemble."""
    ensemble_pred = 0.5 * run_a["y_pred"] + 0.5 * run_b["y_pred"]
    metrics = evaluate_model(run_a["y_true"], ensemble_pred)
    print_metrics(name, metrics)
    return {"metrics": metrics, "y_pred": ensemble_pred}


def train_full_and_predict(*, dataset_name: str, xgb_params, output_path: str):
    """Train on the full training set, predict on test, and export a submission."""
    use_features = dataset_name == "features"
    X_train, y_train, X_test, test_ids = build_training_data(use_features=use_features)

    model = build_xgb_model(**xgb_params)
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_test_pred = np.clip(y_test_pred, 0, 1)

    submission = pd.DataFrame(y_test_pred, columns=y_train.columns)
    submission.insert(0, "ID", test_ids)
    submission["c15"] = 0.0
    submission = submission[SUBMISSION_COLUMNS]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_file, index=False)

    print(f"Submission saved to: {output_file}")
    return model, submission


def ensemble_full_predictions(*, output_path: str):
    """Train two full models and export the mean of their test predictions."""
    _, y_train, _, test_ids = build_training_data(use_features=True)

    _, submission_raw = train_full_and_predict(
        dataset_name="raw",
        xgb_params=BASELINE_PARAMS,
        output_path="data/processed/submission_raw_tmp.csv",
    )
    _, submission_reg = train_full_and_predict(
        dataset_name="features",
        xgb_params=REGULARIZED_PARAMS,
        output_path="data/processed/submission_regularized_tmp.csv",
    )

    pred_columns = list(y_train.columns)
    ensemble = pd.DataFrame()
    ensemble["ID"] = test_ids
    ensemble[pred_columns] = 0.5 * (
        submission_raw[pred_columns].to_numpy() + submission_reg[pred_columns].to_numpy()
    )
    ensemble["c15"] = 0.0
    ensemble = ensemble[SUBMISSION_COLUMNS]

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output_file, index=False)

    print(f"Ensemble submission saved to: {output_file}")
    return ensemble


def build_raw_submission():
    """Train the raw baseline on the full train set and export a submission."""
    return train_full_and_predict(
        dataset_name="raw",
        xgb_params=BASELINE_PARAMS,
        output_path="data/processed/submission_xgboost_raw.csv",
    )


def build_regularized_feature_submission():
    """Train the regularized feature model on the full train set and export a submission."""
    return train_full_and_predict(
        dataset_name="features",
        xgb_params=REGULARIZED_PARAMS,
        output_path="data/processed/submission_xgboost_features_regularized.csv",
    )


def main():
    raw_vs_features = compare_raw_vs_features()
    regularized = run_regularized_feature_model()

    evaluate_ensemble(
        raw_vs_features["raw_humidity"],
        regularized["regularized_features_humidity"],
        name="Ensemble - raw baseline + regularized features - humidity",
    )


if __name__ == "__main__":
    main()
