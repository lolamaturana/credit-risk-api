# Practica 2.2 - Modelización en Ingeniería de Datos 

## Curso 2025-26 <img src="https://images.griddo.cunef.edu/logo-cunef-universidad-1272515f-17b3-4169-8bf6-ef63bfffe920" width="190" valign="middle"> 

### Descripción del Proyecto

Este repositorio contiene el segundo repositorio de la Práctica 2, que consiste en el desarrollo e implementación de una **API REST local utilizando FastAPI y Uvicorn**, gestionada de forma eficiente mediante **`uv`**.

La API está diseñada de manera totalmente independiente al flujo de modelado y su propósito es servir predicciones probabilísticas de impago en tiempo real a partir de datos crudos de clientes. Incorpora **cuantificación de incertidumbre epistémica** mediante calibración con intervalos de Venn-Abers, aplicando de forma estricta una política de negocio para la derivación automatizada o humana de los expedientes de crédito.

---

## Arquitectura y Estructura del Repositorio
El proyecto está organizado como un paquete de Python autónomo bajo la siguiente estructura:

```text
credit-risk-api/
├── api/
│   ├── models/                 
│   │   ├── feature_schema.json # Definición del orden y nombres de las 163 columnas
│   │   ├── filter.pkl          # Pipeline de filtrado entrenado (Práctica 1)
│   │   ├── practica2_model.pkl # Modelo calibrado con Venn-Abers (Repositorio 1 - Práctica 2)
│   │   └── preprocessor.pkl    # Pipeline de preprocesamiento entrenado (Práctica 1)
│   ├── __init__.py             
│   └── main.py                 # Código principal de FastAPI y lógica de endpoints
├── src/                        # Código fuente heredado para la reconstrucción de objetos
│   ├── filtering/
│   └── preprocessing/          # Preprocesador adaptado para flujos streaming/DataFrames
├── .gitignore
├── .python-version
├── ejemplo.txt                 # JSON real con el payload completo de un cliente para pruebas rápidas
├── model_predict.pdf           # Evidencia en PDF de la ejecución exitosa de /predict/
├── model_upload.pdf            # Evidencia en PDF de la ejecución exitosa de /upload_model/
├── pyproject.toml              
├── uv.lock                     
└── README.md                   
```

## Instalación y Despliegue Local

### 1. Arrancar el servidor de desarrollo mediante el comando unificado de `uv`:
```uv run uvicorn api.main:app --reload --port 8080```
Una vez ejecutado, el servicio estará activo en el puerto 8080. Se puede acceder a la interfaz de documentación interactiva automatizada (Swagger UI) en: http://127.0.0.1:8080/docs

### 2. Validar el funcionamiento con el archivo de prueba:
En la raíz del proyecto dispones del archivo `ejemplo.txt`, el cual contiene un payload estructurado en formato JSON con las variables crudas de un cliente real extraído del dataset `accepted_2007_to_2017.csv`. Puedes copiar el contenido de este archivo directamente en el cuadro de texto del endpoint correspondiente en Swagger para verificar que el pipeline procesa los datos sin errores.

---

## Endpoints del Servicio y Ejemplos de Invocación

### 1. Carga Dinámica del Pipeline (`POST /upload_model/`)
Este endpoint multipart/form-data permite subir o actualizar en caliente los tres artefactos binarios (`.pkl`) y el esquema de variables (`.json`) directamente a la memoria RAM de la API sin necesidad de reiniciar el proceso de Uvicorn. Además, implementa una validación técnica que comprueba que el objeto de modelo posea el método `predict_proba`.

* **Invocación mediante `curl`:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8080/upload_model/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'preprocessing=@api/models/preprocessor.pkl' \
  -F 'filtering=@api/models/filter.pkl' \
  -F 'model=@api/models/practica2_model.pkl' \
  -F 'schema=@api/models/feature_schema.json;type=application/json'
```

* **Cuerpo de la Respuesta Exitosa (HTTP 200 OK):**
{
  "status": "ok",
  "message": "Pipeline y esquema actualizados correctamente.",
  "saved": {
    "preprocessing": {
      "path": "/Users/lola/Documents/REPOS_PUBLICADOS/credit-risk-api/api/models/preprocessor.pkl",
      "size": 64414120
    },
    "filtering": {
      "path": "/Users/lola/Documents/REPOS_PUBLICADOS/credit-risk-api/api/models/filter.pkl",
      "size": 6585419
    },
    "model": {
      "path": "/Users/lola/Documents/REPOS_PUBLICADOS/credit-risk-api/api/models/practica2_model.pkl",
      "size": 28081144
    },
    "schema": {
      "path": "/Users/lola/Documents/REPOS_PUBLICADOS/credit-risk-api/api/models/feature_schema.json",
      "size": 6125
    }
  }
}


### 2. Inferencia y Evaluación de Riesgo (POST /predict/)
Recibe los datos en crudo del cliente (acepta tanto un único diccionario como un lote/batch en formato lista). Internamente, la API realiza de forma secuencial la ingesta de las variables, la imputación robusta de nulos, la codificación categórica y de texto adaptada a flujos de streaming, el filtrado/reordenación bajo el esquema exacto del modelo y la predicción multi-fold calibrada.

* **Invocación mediante `curl` (Payload basado en `ejemplo.txt` reducido):**
```bash
curl -X 'POST' \
  '\http://127.0.0.1:8080/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
    "id": 68407277,
    "loan_amnt": 3600.0,
    "term": " 36 months",
    "int_rate": 13.99,
    "emp_title": "leadman",
    "annual_inc": 55000.0,
    "desc": null,
    "dti": 5.91,
    "earliest_cr_line": "Aug-2003",
    "fico_range_low": 675.0,
    "fico_range_high": 679.0
  }'
```

* **Cuerpo de la Respuesta Exitosa (HTTP 200 OK):**
```bash
{
  "p_default": 0.1989,
  "p_low": 0.1529,
  "p_high": 0.2075,
  "decision": "auto",
  "reason": "Incertidumbre controlada."
}
```

## Lógica de Negocio y Política de Derivación (Incertidumbre Epistémica)

La API explota la cuantificación de incertidumbre derivada de los límites probabilísticos calculados mediante el calibrador de Venn-Abers. La métrica utilizada para activar las reglas automatizadas es la anchura del intervalo:
$$\text{Anchura del Intervalo} = p_{\text{high}} - p_{\text{low}}$$

* Procesamiento Automático ("decision": "auto"): Ocurre cuando la anchura es $\le 0.2$. El modelo cuenta con suficiente densidad de datos históricos similares a los del cliente evaluado como para confiar plenamente en la probabilidad puntual otorgada.

* Derivación a Analista ("decision": "agent"): Se activa si la anchura del intervalo es $> 0.2$. El sistema identifica que el cliente presenta características atípicas que inducen una alta incertidumbre epistémica, derivando el expediente a un agente humano especializado para su evaluación manual.
