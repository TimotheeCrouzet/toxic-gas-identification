import pandas as pd
from sklearn.model_selection import train_test_split

from challenge_ml.feature import build_training_data


def random_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and targets randomly into train and validation sets."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_valid, y_train, y_valid


def humidity_split(
    X: pd.DataFrame,
    y: pd.DataFrame,
    humidity_col: str = "Humidity",
    humidity_quantile: float = 0.8,
):
    """Split using the highest humidity rows as validation data."""
    humidity = X[humidity_col]
    humidity_threshold = humidity.quantile(humidity_quantile)
    humidity_shift_mask = humidity >= humidity_threshold

    X_train = X.loc[~humidity_shift_mask].copy()
    X_valid = X.loc[humidity_shift_mask].copy()
    y_train = y.loc[~humidity_shift_mask].copy()
    y_valid = y.loc[humidity_shift_mask].copy()

    return X_train, X_valid, y_train, y_valid


def random_split_from_features(
    test_size: float = 0.2,
    random_state: int = 42,
    use_features: bool = True,
):
    """Load prepared data, then apply a random split."""
    X, y, _, _ = build_training_data(use_features=use_features)
    return random_split(X, y, test_size=test_size, random_state=random_state)


def humidity_split_from_features(
    humidity_quantile: float = 0.8,
    use_features: bool = True,
):
    """Load prepared data, then apply a humidity-based split."""
    X, y, _, _ = build_training_data(use_features=use_features)
    return humidity_split(X, y, humidity_quantile=humidity_quantile)
