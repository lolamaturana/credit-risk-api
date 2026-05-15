import pickle
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from datetime import datetime

app = FastAPI(
    title="API de Riesgo de Crédito (Venn-Abers)",
    description="API REST para predecir impago con cuantificación de incertidumbre.",
    version="1.0"
)

MODELS_DIR = Path(__file__).parent / "models"
ARTIFACT_FILENAMES = {
    "preprocessing": "preprocessor.pkl",
    "filtering": "filter.pkl",
    "model": "practica2_model.pkl",
    "schema": "feature_schema.json"
}

def _load_artifact(filename: str) -> Any:
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    try:
        if filename.endswith(".json"):
            with path.open("r") as f:
                return json.load(f)
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[warn] no se pudo cargar {path.name}: {e}")
        return None

# Carga inicial de variables globales
preprocessor = _load_artifact(ARTIFACT_FILENAMES["preprocessing"])
feature_filter = _load_artifact(ARTIFACT_FILENAMES["filtering"])
model = _load_artifact(ARTIFACT_FILENAMES["model"])
feature_schema = _load_artifact(ARTIFACT_FILENAMES["schema"])

@app.post("/predict/")
async def predict(data: dict[str, Any] | list[dict[str, Any]]):
    if preprocessor is None or feature_filter is None or model is None or feature_schema is None:
        raise HTTPException(
            status_code=503,
            detail="Modelos no cargados o esquema faltante. Usa /upload_model/.",
        )

    is_batch = isinstance(data, list)
    df_raw = pd.DataFrame(data if is_batch else [data])

    try:
        # 1. Preprocesamiento
        df_pre, _ = preprocessor.transform(df_raw)

        # 2. Filtrado y creación de DataFrame con nombres de columnas
        X_filt = feature_filter.transform(df_pre)
        columnas_esperadas = feature_schema["features"]
        df_filt = pd.DataFrame(X_filt, columns=columnas_esperadas)

        # 3. Reordenar (por si el filtro movió algo)
        df_final = df_filt[columnas_esperadas]

        # 4. Predicción con Venn-Abers
        probs_va, p0p1 = model.predict_proba(df_final, p0_p1_output=True)
        p0p1 = np.asarray(p0p1)

        # Lógica para extraer p_low y p_high (manejando CVAP o Inductive)
        if p0p1.ndim == 3:
            p_low_arr = p0p1[:, :, 0].mean(axis=0)
            p_high_arr = p0p1[:, :, 1].mean(axis=0)
        else:
            p_low_arr = p0p1[:, 0]
            p_high_arr = p0p1[:, 1]
            
        p_default_arr = probs_va[:, 1] if probs_va.ndim == 2 else probs_va

        # 5. Formatear la salida para la rúbrica
        rows = []
        for i in range(len(df_final)):
            p_default = float(p_default_arr[i])
            p_low = float(p_low_arr[i])
            p_high = float(p_high_arr[i])
            width = p_high - p_low
            
            decision = "agent" if width > 0.2 else "auto"
            reason = f"p_high - p_low ({width:.4f}) > 0.2" if width > 0.2 else "Incertidumbre controlada."
                
            rows.append({
                "p_default": round(p_default, 4),
                "p_low": round(p_low, 4),
                "p_high": round(p_high, 4),
                "decision": decision,
                "reason": reason
            })

        return rows if is_batch else rows[0]

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error en el procesamiento: {str(e)}")

@app.post("/upload_model/")
async def upload_model(
    preprocessing: UploadFile = File(...),
    filtering: UploadFile = File(...),
    model_file: UploadFile = File(..., alias="model"),
    schema_file: UploadFile = File(..., alias="schema"), # Añadido el esquema
):
    global model, preprocessor, feature_filter, feature_schema

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    uploads = {
        "preprocessing": preprocessing,
        "filtering": filtering,
        "model": model_file,
        "schema": schema_file,
    }
    
    saved = {}
    loaded = {}

    for key, upload in uploads.items():
        # Validar extensiones
        ext = ".json" if key == "schema" else ".pkl"
        if not upload.filename.endswith(ext):
            raise HTTPException(
                status_code=400,
                detail=f"'{key}' debe ser un fichero {ext} (recibido: {upload.filename})",
            )
        
        destination = MODELS_DIR / ARTIFACT_FILENAMES[key]
        contents = await upload.read()
        destination.write_bytes(contents)
        
        # Cargar en memoria según el tipo de archivo
        if key == "schema":
            loaded[key] = json.loads(contents)
        else:
            loaded[key] = pickle.loads(contents)
            
        saved[key] = {"path": str(destination), "size": len(contents)}

    # Validación técnica de la rúbrica
    if not hasattr(loaded["model"], "predict_proba"):
         raise HTTPException(status_code=400, detail="El modelo no posee el método predict_proba.")

    # Actualizar las variables globales con los nuevos objetos cargados
    preprocessor = loaded["preprocessing"]
    feature_filter = loaded["filtering"]
    model = loaded["model"]
    feature_schema = loaded["schema"]

    return {
        "status": "ok", 
        "message": "Pipeline y esquema actualizados correctamente.",
        "saved": saved,
        "timestamp": datetime.now().isoformat()
    }