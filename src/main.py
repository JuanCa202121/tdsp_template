
# ----------------------------- main.py -----------------------------
"""Punto de entrada del pipeline (train, eval, deploy, serve)."""
import argparse
from preprocessing import load_data, clean_raw, split_xy, train_test_split_custom
from training import train_and_tune, save_model
from evaluation import evaluate
from deployment import log_model_mlflow, serve_model_mlflow


def main():
    parser = argparse.ArgumentParser(description="Pipeline XGBoost")
    parser.add_argument("mode", choices=["train", "eval", "deploy", "serve"], help="Modo de ejecución")
    parser.add_argument("--model-uri", default=None, help="URI del modelo a servir (opcional)")
    args = parser.parse_args()

    if args.mode in {"train", "eval"}:
        df = clean_raw(load_data())
        X, y = split_xy(df)
        X_train, X_test, y_train, y_test = train_test_split_custom(X, y)

    if args.mode == "train":
        best_model = train_and_tune(X_train, y_train)
        save_model(best_model)
    elif args.mode == "eval":
        evaluate("xgb_credit_default.pkl", X_test, y_test)
    elif args.mode == "deploy":
        log_model_mlflow("xgb_credit_default.pkl", "XGBoost_Optimizado")
    elif args.mode == "serve":
        serve_model_mlflow(args.model_uri)
    else:
        print("Modo no reconocido")

if __name__ == "__main__":
    main()

