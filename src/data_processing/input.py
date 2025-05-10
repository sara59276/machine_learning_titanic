import pandas as pd
from sklearn.model_selection import train_test_split

from constants.constants import PROJECT_ROOT


TRAIN_DATA_PATH = f"{PROJECT_ROOT}/resource/train.csv"
EVAL_DATA_PATH = f"{PROJECT_ROOT}/resource/eval.csv"

IDS = [
    "PassengerId",
]
TARGET = [
    "Survived",
]
ALL_FEATURES = [
    "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin",
    "Embarked_S", "Embarked_C", "Embarked_Q",
]
SELECTED_FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked_S", "Embarked_C", "Embarked_Q",
]
DROPPED_FEATURES = [
    "Name", "Ticket", "Cabin",
]


def _load_raw_data(path: str):
    return pd.read_csv(
        path,
        converters={'Sex': lambda x: 1 if x == 'female' else 0},
    )


def _process_data(df: pd.DataFrame):
    # fill missing ages with median
    df["Age"] = df["Age"].fillna(df["Age"].median())  # TODO change to regression-based imputation

    # one-hot encoding for "Embarked" feature
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)
    df['Embarked_C'] = (df['Embarked'] == 'C').astype(int)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)

    return df


def get_training_data(selected_features_only: bool = False):
    df = _load_raw_data(TRAIN_DATA_PATH)
    df = _process_data(df)
    X = df[SELECTED_FEATURES] if selected_features_only else df[ALL_FEATURES]
    y = df[TARGET].astype(int)
    ids = df[IDS[0]]
    return X, y, ids


def get_training_data_split(test_size: float, features_only: bool = False):
    X, y, ids = get_training_data(selected_features_only=features_only)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
    return X_train, X_test, y_train, y_test, ids


def get_evaluation_data(selected_features_only: bool = False):
    df = _load_raw_data(EVAL_DATA_PATH)
    df = _process_data(df)
    X = df[SELECTED_FEATURES] if selected_features_only else df[ALL_FEATURES]
    ids = df[IDS[0]]
    return X, ids
