# 🔎 Detección de Anomalías en Consumo de Gas con Isolation Forest (Azure ML + Streamlit)

## 📌 Descripción del Proyecto
Este proyecto implementa un **sistema de detección de anomalías en series temporales de consumo de gas** como parte de un microproyecto académico en la nube (Azure Machine Learning).  
El flujo completo cubre desde la **carga de datos en Excel/CSV**, el **entrenamiento de modelos con Isolation Forest**, el **registro y despliegue automático de modelos en un endpoint gestionado de Azure ML**, hasta la **inferencia en línea con visualización interactiva de anomalías** vía **Streamlit**.

La solución permite un ciclo completo de **carga de datos → entrenamiento → despliegue → inferencia → descarga de resultados → visualización interactiva**, con capacidad de **actualizar versiones de modelos** de forma continua en Azure.

---

## 📂 Estructura del Proyecto
```bash
anom-detector/
├── .venv/                  # Entorno virtual local
├── aml/                    # Configuración y scripts para Azure ML
│   ├── env/
│   │   └── conda.yml        # Dependencias Conda para entrenar en Azure
│   ├── inference/
│   │   ├── score.py         # Script de inferencia (modelo desplegado en endpoint)
│   │   ├── job-train.yml    # Definición de Job de entrenamiento en Azure ML
│   │   ├── online-endpoint.yml # Configuración de endpoint en Azure ML
│   │   ├── online-deployment.yml # Configuración de deployment en Azure ML
│   └── train.py             # Script de entrenamiento en Azure ML
├── .env                     # Variables de entorno locales (endpoint URL, key)
├── .env.example             # Ejemplo de configuración .env
├── app.py                   # Aplicación principal Streamlit (UI)
├── blue-ds3.yml             # Ejemplo de deployment con DS3_v2
├── cpu-spot.yml             # Ejemplo de deployment en spot instances
├── payload.json             # Ejemplo de payload para pruebas
├── resp.json                # Ejemplo de respuesta del endpoint
├── requirements.txt         # Dependencias para la app local
```

---

## ⚙️ Pipeline de Flujo
1. **Carga de datos**: El usuario sube un archivo Excel/CSV desde la UI.
2. **Entrenamiento (Azure Job)**:
   - Se crea un **Data Asset** en Azure ML.
   - Se lanza un **job de entrenamiento** (`aml/train.py`) que genera un modelo Isolation Forest.
   - El modelo y metadatos se guardan en `outputs/` y se registran en el workspace.
3. **Registro de modelo**: El job registrado se convierte en un modelo versionado en Azure ML.
4. **Despliegue de endpoint**:
   - Se crea/actualiza un **deployment online** en un endpoint gestionado de Azure ML.
   - Se ajusta a un `instance_type` adecuado (`Standard_DS3_v2` recomendado).
5. **Inferencia en línea**:
   - El usuario sube datos nuevos en la UI.
   - Se construye un `payload.json` y se invoca el endpoint vía:
     - `az ml online-endpoint invoke` (recomendado)
     - `az rest`
     - o **Requests Python** (fallback).
6. **Visualización y descarga**:
   - La app muestra las anomalías en gráficos Plotly interactivos.
   - Se ofrece la descarga de resultados en CSV (`;` como separador).
   - Gráficos disponibles:
     - Serie completa con anomalías resaltadas.
     - Solo anomalías en zoom.

---

## 🛠️ Preparación del entorno local (Windows, **venv**)
> Si ya tienes tu entorno listo, puedes saltar a “Paso a Paso de Montaje”.

1. Verifica Python:
   ```bat
   python --version
   ```
2. Crear el **entorno virtual** en la carpeta del proyecto:
   ```bat
   python -m venv .venv
   ```
3. **Activar** el entorno:
   ```bat
   .venv\Scripts\activate
   ```
4. Actualizar `pip` e instalar dependencias:
   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Copiar variables de entorno y completarlas:
   ```bat
   copy .env.example .env
   ```
   Edita `.env` con tu `AML_ENDPOINT_URL` y `AML_ENDPOINT_KEY`.
6. (Opcional) **Desactivar** el entorno cuando termines:
   ```bat
   deactivate
   ```

---

## 🚀 Paso a Paso de Montaje

### 1. Crear el **Resource Group** en Azure
```bash
az group create --name ws-gas-anom-wus3 --location westus3
```

### 2. Crear el **Workspace de Azure ML**
```bash
az ml workspace create --name ws-gas-anom-wus3 --resource-group ws-gas-anom-wus3 --location westus3
```

### 3. Crear el **Compute Cluster**
Ejemplo CPU básico:
```bash
az ml compute create -n cpu-ds1 --type AmlCompute --min-instances 0 --max-instances 2 --size Standard_DS2_v2
```

### 4. Registrar el **Data Asset**
Cuando se sube un dataset en la UI (app.py), este se registra automáticamente como:
```
azureml:gasdata-<UUID>@latest
```

### 5. Lanzar **Job de Entrenamiento**
```bash
az ml job create -f aml/job-train.yml
```

### 6. Registrar el **Modelo**
```bash
az ml model create --name iforest-gas --type custom_model --path azureml://jobs/<JOB_ID>/outputs/artifacts/paths/outputs/model_iforest.pkl
```

### 7. Crear el **Endpoint Online**
```bash
az ml online-endpoint create -f aml/online-endpoint.yml
```

### 8. Crear el **Deployment**
```bash
az ml online-deployment create -f aml/online-deployment.yml --all-traffic
```

> ⚠️ Importante: usar `Standard_DS3_v2` o superior para evitar errores de memoria.

### 9. Configurar **.env**
```env
AML_ENDPOINT_URL=https://gas-iforest-endpoint.westus3.inference.ml.azure.com/score
AML_ENDPOINT_KEY=<tu_primary_key>
AML_ENDPOINT_NAME=gas-iforest-endpoint
AML_DEPLOYMENT_NAME=blue
```

### 10. Ejecutar la **App Streamlit**
> Asegúrate de tener el **venv activado** (`.venv\Scripts\activate`) antes de correr:
```bash
streamlit run app.py
```

---

## 🎛️ Uso de la Interfaz (Streamlit)
1. **Entrenar modelo**:
   - Subir CSV/Excel.
   - Especificar columna de fecha (`FechaHora`) y variable (`VolCorrected`).
   - Definir `contamination`.
   - Ejecutar entrenamiento y despliegue automático en Azure.
2. **Detectar anomalías**:
   - Subir nuevo CSV/Excel.
   - Invocar endpoint (CLI o REST).
   - Ver tabla de resultados, descargar CSV.
   - Explorar visualizaciones:
     - Serie completa con anomalías resaltadas.
     - Solo anomalías (zoom).
3. **Opciones avanzadas**:
   - Editar endpoint, deployment, suscripción, grupo y workspace desde la barra lateral.
   - Guardar configuración en `.env`.

---

## 📊 Ejemplo de Respuesta del Endpoint
Request:
```json
{"data":[{"FechaHora":"2024-01-01 00:00:00","VolCorrected":1.23}]}
```

Response:
```json
{
  "ok": true,
  "rows": 1,
  "preview": [
    {"FechaHora": "2024-01-01 00:00:00", "VolCorrected": 1.23, "anomaly": 1}
  ],
  "csv_base64": "..."
}
```

---

## 🔍 Ver **servicios activos** (endpoints/deployments) en Azure ML
> Ejecuta en **CMD/PowerShell** (con `az login` hecho y, si aplica, `az account set`):

- **Listar endpoints online del workspace** (tabla):
  ```bat
  az ml online-endpoint list -o table
  ```
- **Ver detalles de tu endpoint** (JSON coloreado):
  ```bat
  az ml online-endpoint show -n gas-iforest-endpoint -o jsonc
  ```
- **Listar deployments del endpoint** (tabla):
  ```bat
  az ml online-deployment list --endpoint-name gas-iforest-endpoint -o table
  ```

---

## ✅ Tecnologías Utilizadas
- **Azure Machine Learning** (Jobs, Data Assets, Endpoints, Deployments)
- **Python** (scikit-learn, pandas, numpy, joblib)
- **Streamlit** (UI)
- **Plotly** (visualización)
- **Azure CLI** (automatización)
- **dotenv** (manejo de credenciales)
