    
# ----------------------------- evaluation.py -----------------------------
"""Funciones de evaluación y visualización."""
import joblib, os
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["evaluate"]

def evaluate(model_path, X_test, y_test):
    print(f"🔍  Evaluando modelo: {model_path}")
    model = joblib.load(model_path)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}"); plt.plot([0,1],[0,1],'k--'); plt.title("Curva ROC"); plt.legend(); plt.savefig("roc.png"); print("📈  ROC guardada en roc.png")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1]); plt.title("Matriz de Confusión"); plt.savefig("cm.png"); print("📈  CM guardada en cm.png")
