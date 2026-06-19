import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)  # type: ignore
    df = df.rename(columns=str.lower)
    df = df.sort_values("ts")  # type: ignore
    df = df.drop_duplicates(subset=["ts"])
    return df


def time_split(X, y, n_splits=5):  # type: ignore
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):  # type: ignore
        yield X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]  # type: ignore
