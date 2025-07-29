import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "load_data",
    "clean_raw",
    "split_xy",
    "train_test_split_custom",
]

def load_data(path: str = "./data/UCI_Credit_Card.csv") -> pd.DataFrame:
    """Carga el dataset CSV y renombra la columna objetivo."""
    df = pd.read_csv(path)
    return df.rename(columns={"default.payment.next.month": "target"})


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """limpieza y selección de categorías."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype("category")
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3}).astype("category")
    df["SEX"] = df["SEX"].astype("category")
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["ID", "target"])
    y = df["target"]
    return X, y


def train_test_split_custom(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)