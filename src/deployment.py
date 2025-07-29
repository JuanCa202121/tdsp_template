# ----------------------------- deployment.py -----------------------------
"""Registro y serving del modelo con MLflow y Model Registry."""
import mlflow
from mlflow.tracking import MlflowClient
import joblib, os

__all__ = ["log_model_mlflow", "serve_model_mlflow"]

def log_model_mlflow(model_path: str, model_name: str):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    mlruns_path = os.path.join(root_dir, "mlruns")
    mlflow.set_tracking_uri(f"file:///{mlruns_path}")
    mlflow.set_experiment("Entrega_Deployment")

    # Cargar el modelo y logear como artefacto
    model = joblib.load(model_path)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, model_name)
        run_id = run.info.run_id
        print(f"🚀  Modelo registrado en MLflow (run_id={run_id})")

    # Registrar en el Model Registry
    client = MlflowClient(tracking_uri=f"file:///{mlruns_path}")
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # ya existe
    model_uri = f"runs:/{run_id}/{model_name}"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    print(f"📦  Version {mv.version} de '{model_name}' creada en el Registry")


def serve_model_mlflow(model_uri: str = None, port: int = 5001):
    # Default: usar última versión del modelo registrado
    if model_uri is None:
        model_uri = "models:/XGBoost_Optimizado/1"
    print(f"🌐  Sirviendo modelo en {model_uri} -> http://localhost:{port}/invocations …")
    os.system(
        f"mlflow models serve --model-uri \"{model_uri}\" --host 0.0.0.0 --port {port} --workers 2 --no-conda"
    )
