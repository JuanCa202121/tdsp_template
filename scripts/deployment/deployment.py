import mlflow, os, joblib

__all__ = [
    "log_model_mlflow",
    "serve_model_mlflow",
]

def log_model_mlflow(model_path: str, model_name: str):
    # Definir ruta absoluta al directorio raíz del proyecto
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = os.path.join(root_dir, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment("Entrega_Deployment")
    model = joblib.load(model_path)
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, model_name)

def serve_model_mlflow(model_uri: str, port: int = 5001):
    os.system(
        f"mlflow models serve --model-uri \"{model_uri}\" --host 0.0.0.0 --port {port} --workers 2 --no-conda"
    )