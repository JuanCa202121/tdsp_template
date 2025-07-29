def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "deploy", "serve"], help="Modo de ejecución")
    args = parser.parse_args()

    df = load_data()
    df_clean = clean_raw(df)
    X, y = split_xy(df_clean)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y)

    if args.mode == "train":
        best_model = train_and_tune(X_train, y_train)
        save_model(best_model)
        print("Modelo entrenado y guardado.")
    elif args.mode == "eval":
        evaluate("xgb_credit_default.pkl", X_test, y_test)
    elif args.mode == "deploy":
        log_model_mlflow("xgb_credit_default.pkl", "XGBoost_Optimizado")
    elif args.mode == "serve":
        serve_model_mlflow("models:/XGBoost_Optimizado/2")
    else:
        print("Modo no reconocido")

if __name__ == "__main__":
    main()