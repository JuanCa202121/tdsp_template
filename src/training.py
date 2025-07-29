# ----------------------------- training.py -----------------------------
"""Entrenamiento y optimización del modelo XGBoost."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import joblib, os, time

__all__ = [
    "build_preprocessor",
    "build_model",
    "train_and_tune",
    "save_model",
]

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include="category").columns.tolist()
    num_cols = X.select_dtypes(exclude="category").columns.tolist()
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    
def build_model(scale_pos_weight: float):
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    
def train_and_tune(X_train, y_train):
    if os.path.exists("xgb_credit_default.pkl"):
        print("📂  Modelo existente encontrado → cargando …")
        return joblib.load("xgb_credit_default.pkl")

    print("🔄  Entrenando modelo XGBoost con búsqueda de hiperparámetros …")
    t0 = time.time()
    preprocess = build_preprocessor(X_train)
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = build_model(ratio)
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model),
    ])
    param_grid = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [3, 4],
        "model__learning_rate": [0.01, 0.05],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"✅  Entrenamiento finalizado ({time.time()-t0:.1f}s). Mejor AUC: {grid.best_score_:.4f}")
    return grid.best_estimator_

def save_model(model, path: str = "xgb_credit_default.pkl"):
    joblib.dump(model, path)
    print(f"💾  Modelo guardado en {path}")
