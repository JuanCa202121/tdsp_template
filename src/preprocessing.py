# ----------------------------- preprocessing.py -----------------------------
"""Preprocessing utilities: carga, limpieza y división de datos.
Dataset original: data/UCI_Credit_Card.csv"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = [
    "load_data",
    "clean_raw",
    "split_xy",
    "train_test_split_custom",
]

def load_data(path: str = None) -> pd.DataFrame:
    # Determinar ruta al dataset desde la raíz del proyecto
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = path or os.path.join(root_dir, "data", "UCI_Credit_Card.csv")
    print(f"📥  Cargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    df.rename(columns={"default.payment.next.month": "target"}, inplace=True)
    print(f"✅  Filas cargadas: {len(df)}")
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹  Limpiando datos …")
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}).astype("category")
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3}).astype("category")
    df["SEX"]       = df["SEX"].astype("category")
    print("✅  Limpieza terminada")
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=["ID", "target"])
    y = df["target"]
    return X, y


def train_test_split_custom(X, y, test_size: float = 0.2, random_state: int = 42):
    print("✂️   Dividiendo en train/test …")
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)