from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-6
RATIO_EPSILON = 1e-3
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def signed_log1p(values: pd.Series) -> pd.Series:
    return np.sign(values) * np.log1p(np.abs(values))


def stable_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    raw_ratio = numerator / (np.abs(denominator) + RATIO_EPSILON)
    return signed_log1p(raw_ratio)


def feature_engineering(features: pd.DataFrame) -> pd.DataFrame:
    engineered = features.copy()

    engineered["M_high_mean"] = engineered[["M12", "M13", "M14", "M15"]].mean(axis=1)
    engineered["M_low_mean"] = engineered[["M4", "M5", "M6", "M7"]].mean(axis=1)

    engineered["M_high_std"] = engineered[["M12", "M13", "M14", "M15"]].std(axis=1)
    engineered["M_low_std"] = engineered[["M4", "M5", "M6", "M7"]].std(axis=1)

    engineered["M_high_max"] = engineered[["M12", "M13", "M14", "M15"]].max(axis=1)
    engineered["M_low_max"] = engineered[["M4", "M5", "M6", "M7"]].max(axis=1)

    engineered["M4_M5_ratio"] = stable_ratio(engineered["M4"], engineered["M5"])
    engineered["M6_M7_ratio"] = stable_ratio(engineered["M6"], engineered["M7"])
    engineered["M_ratio"] = stable_ratio(
        engineered["M_low_mean"], engineered["M_high_mean"]
    )

    engineered["M4_H"] = engineered["M4"] * engineered["Humidity"]
    engineered["M_mean_H"] = engineered["M_low_mean"] * engineered["Humidity"]

    for col in ["M4", "M5", "M6", "R", "S2"]:
        engineered[f"log_{col}"] = np.log1p(np.abs(engineered[col]))

    sensor_cols = ["M4", "M5", "M6", "M7", "M12", "M13", "M14", "M15"]
    engineered["sensor_var"] = engineered[sensor_cols].var(axis=1)

    engineered["S_mean"] = engineered[["S1", "S2", "S3"]].mean(axis=1)
    engineered["R_S_ratio"] = engineered["R"] / (engineered["S_mean"] + EPSILON)

    return engineered


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = pd.read_csv(DATA_DIR / "x_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    x_test = pd.read_csv(DATA_DIR / "x_test.csv")
    return x_train, y_train, x_test


def build_training_data(
    use_features: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train, y_train, x_test = load_datasets()

    test_ids = x_test["ID"].copy()

    x_train = x_train.drop(columns=["ID"])
    x_test = x_test.drop(columns=["ID"])
    if use_features:
        x_train = feature_engineering(x_train)
        x_test = feature_engineering(x_test)
    y_train = y_train.drop(columns=["ID", "c15"])

    return x_train, y_train, x_test, test_ids


if __name__ == "__main__":
    x_train, y_train, x_test, test_ids = build_training_data()
    print(x_train.shape, y_train.shape, x_test.shape, test_ids.shape)
