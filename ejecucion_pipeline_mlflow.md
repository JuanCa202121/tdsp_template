# 🧪 Comandos para ejecutar el pipeline completo, servir el modelo y hacer inferencia

### 1. Activar el entorno conda
```bash
conda activate deepdl
```

---

### 2. Ejecutar el entrenamiento (entrena y guarda el modelo)
```bash
python src/main.py train
```

---

### 3. Evaluar el modelo (muestra métricas y guarda gráficas)
```bash
python src/main.py eval
```

---

### 4. Registrar el modelo en MLflow Model Registry
```bash
python src/main.py deploy
```

---

### 5. Servir el modelo localmente (crea endpoint local con MLflow)
```bash
python src/main.py serve
```

Este comando inicia el servidor en:
```
http://localhost:5001/invocations
```

Dejar el servidor corriendo.

---

### 6. En una nueva terminal: activar ngrok (exponer el endpoint)
```bash
ngrok authtoken TU_TOKEN
ngrok http 5001
```

Esto te dará una URL pública como:
```
https://abcd1234.ngrok-free.app → http://localhost:5001
```

---

### 7. En otra terminal o herramienta (como Postman): hacer inferencia
Crear un archivo llamado `payload.json` con datos reales o simulados de entrada:

```json
{
  "columns": ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"],
  "data": [[50000, 2, 2, 1, 25, 0, 0, 0, 0, 0, 0, 3913, 3102, 689, 0, 0, 0, 0, 689, 0, 0, 0, 0]]
}
```

Y luego correr en terminal:

```bash
curl -X POST https://abcd1234.ngrok-free.app/invocations \
     -H "Content-Type: application/json" \
     -d @payload.json
```

Te devolverá una predicción como:
```json
{"predictions": [0]}
```

---

> Reemplaza `https://abcd1234.ngrok-free.app` por la URL que te dé tu terminal al ejecutar `ngrok http 5001`.
