# app.py
import os
import io
import json
import base64
import time
import uuid
import shutil
import tempfile
import subprocess
import platform
from typing import Tuple, Optional, Any

import pandas as pd
import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import shutil as _shutil
import urllib3
import ssl
from urllib3.util.ssl_ import create_urllib3_context
from requests.adapters import HTTPAdapter

load_dotenv()

# --- FIX: helper para “desenvolver” respuestas anidadas (result/output/body/response/value)
def _unwrap_response(x):
    """Desanida respuestas Azure/CLI bajo result/output/body/response/value.
    Si detecta 'csv_base64' sin 'ok', infiere ok=True para compatibilidad."""
    def _to_json(maybe_json):
        if isinstance(maybe_json, str):
            s = maybe_json.strip()
            try:
                return json.loads(s)
            except Exception:
                return maybe_json
        return maybe_json
    x = _to_json(x)
    for _ in range(4):
        if isinstance(x, dict):
            for k in ("result", "output", "body", "response", "value"):
                if k in x:
                    x = _to_json(x[k])
                    break
            else:
                break
        else:
            break
    if isinstance(x, dict) and ("csv_base64" in x) and ("ok" not in x):
        x["ok"] = True
    return x if isinstance(x, dict) else {"result": x}

# =========================
# Config inicial
# =========================
AML_URL = os.getenv("AML_URL") or os.getenv("AML_ENDPOINT_URL", "")
AML_KEY = os.getenv("AML_KEY") or os.getenv("AML_ENDPOINT_KEY", "")

# Contexto de Azure (para 'az ml online-endpoint invoke')
AZ_SUBSCRIPTION_ID = os.getenv("AZ_SUBSCRIPTION_ID", "")
AZ_RESOURCE_GROUP  = os.getenv("AZ_RESOURCE_GROUP", "")
AZ_ML_WORKSPACE    = os.getenv("AZ_ML_WORKSPACE", "")

DATE_COL_DEFAULT = "FechaHora"
TARGET_COL_DEFAULT = "VolCorrected"
DEFAULT_ENDPOINT_NAME = os.getenv("AML_ENDPOINT_NAME", "gas-iforest-endpoint")
DEFAULT_DEPLOYMENT_NAME = os.getenv("AML_DEPLOYMENT_NAME", "blue")

# --- alias y normalización de encabezados ---
ALIASES = {
    "Hourly_Date": DATE_COL_DEFAULT,
    "Fecha_Hora": DATE_COL_DEFAULT,
    "Date": DATE_COL_DEFAULT,
    "Value": TARGET_COL_DEFAULT,
    "Caudal": TARGET_COL_DEFAULT,
    "Caudal_m3h": TARGET_COL_DEFAULT,
}

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [c.strip() for c in df2.columns]
    for k, v in ALIASES.items():
        if k in df2.columns and v not in df2.columns:
            df2 = df2.rename(columns={k: v})
    return df2

# --- normalización estricta de URL (añade https:// y /score) ---
def _normalize_url(u: str) -> str:
    u = (u or "").strip().strip("'").strip('"')
    if not u:
        return u
    if u.startswith("//"):
        u = "https:" + u
    if not u.lower().startswith("http"):
        u = "https://" + u
    if not u.rstrip("/").endswith("/score"):
        u = u.rstrip("/") + "/score"
    return u

# =========================
# Azure CLI helpers
# =========================
_AZ_BIN_CACHE = None

def _iter_common_az_locations():
    system = platform.system().lower()
    if system == "windows":
        yield os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                           "Microsoft SDKs", "Azure", "CLI2", "wbin", "az.cmd")
        yield os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft SDKs", "Azure", "CLI2", "wbin", "az.cmd")
        yield os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\%USERNAME%\AppData\Local"),
                           "Microsoft", "WindowsApps", "az.cmd")
        yield os.path.join(os.environ.get("LOCALAPPDATA", r"C:\Users\%USERNAME%\AppData\Local"),
                           "Microsoft", "WindowsApps", "az.exe")
    else:
        for p in ("/usr/bin/az", "/usr/local/bin/az", "/opt/homebrew/bin/az", "/home/linuxbrew/.linuxbrew/bin/az"):
            yield p

def _resolve_az_path() -> str:
    global _AZ_BIN_CACHE
    if _AZ_BIN_CACHE and os.path.exists(_AZ_BIN_CACHE):
        return _AZ_BIN_CACHE
    env_hint = os.getenv("AZ_CLI")
    if env_hint and os.path.exists(env_hint):
        _AZ_BIN_CACHE = env_hint
        return _AZ_BIN_CACHE
    for cand in ("az", "az.cmd", "az.exe"):
        p = _shutil.which(cand)
        if p:
            _AZ_BIN_CACHE = p
            return _AZ_BIN_CACHE
    for p in _iter_common_az_locations():
        if os.path.exists(p):
            _AZ_BIN_CACHE = p
            return _AZ_BIN_CACHE
    raise RuntimeError("No se encontró Azure CLI ('az'). Instálala: https://aka.ms/installazurecli")

def _az(*args: str) -> Tuple[int, str, str]:
    az_path = _resolve_az_path()
    proc = subprocess.Popen([az_path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out.strip(), err.strip()

# =========================
# Azure ML helpers (jobs/model/deploy)
# =========================
def az_check_compute_exists(compute_name: str) -> bool:
    rc, _, _ = _az("ml", "compute", "show", "--name", compute_name, "-o", "none")
    return rc == 0

def az_wait_data_asset(name: str, version_or_label: str = "latest", timeout: int = 90, poll_secs: int = 3) -> bool:
    import time as _t
    start = _t.time()
    while _t.time() - start < timeout:
        rc, _, _ = _az("ml", "data", "show", "--name", name, "--version", str(version_or_label), "-o", "none")
        if rc == 0:
            return True
        rc2, out2, _ = _az("ml", "data", "list", "--name", name, "-o", "json")
        if rc2 == 0:
            try:
                items = json.loads(out2)
                if isinstance(items, list) and len(items) > 0:
                    return True
            except Exception:
                pass
        _t.sleep(poll_secs)
    return False

def az_job_train(local_data_path: str,
                 date_col: str,
                 target_col: str,
                 contamination: float,
                 compute: str = "cpu-ds1") -> Optional[str]:
    if not az_check_compute_exists(compute):
        st.error(f"El compute '{compute}' no existe en el workspace actual.")
        return None
    tmp_dir = tempfile.mkdtemp(prefix="upl_")
    abs_src = os.path.abspath(local_data_path)
    dst = os.path.join(tmp_dir, os.path.basename(abs_src))
    shutil.copyfile(abs_src, dst)
    asset_name = f"gasdata-{uuid.uuid4().hex[:10]}"
    rc, out, err = _az("ml", "data", "create", "--name", asset_name, "--type", "uri_folder", "--path", tmp_dir, "-o", "json")
    if rc != 0:
        st.error(f"Error creando Data Asset:\n{err}\n{out}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    asset_ref = f"azureml:{asset_name}@latest"
    if not az_wait_data_asset(asset_name, "latest", timeout=90, poll_secs=3):
        st.error(f"El Data Asset {asset_name}@latest no es visible aún.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None
    cmd = [
        "ml", "job", "create",
        "-f", "aml/job-train.yml",
        "--set", f"inputs.data.path={asset_ref}",
        "--set", f"inputs.date_col={date_col}",
        "--set", f"inputs.target_col={target_col}",
        "--set", f"inputs.contamination={contamination}",
        "--set", f"compute={compute}",
        "-o", "json",
    ]
    rc, out, err = _az(*cmd)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if rc != 0:
        st.error(f"Error creando Job:\n{err}\n{out}")
        return None
    try:
        return json.loads(out).get("name")
    except Exception:
        return None

def az_job_wait(job_id: str, timeout: int = 1800, poll_secs: int = 10) -> Optional[str]:
    start = time.time()
    while time.time() - start < timeout:
        rc, out, err = _az("ml", "job", "show", "--name", job_id, "-o", "json")
        if rc != 0:
            st.error(f"Error consultando job:\n{err}\n{out}")
            return None
        info = json.loads(out)
        status = info.get("status", "")
        st.write(f"Estado: **{status}**")
        if status in {"Completed", "Failed", "Canceled", "Cancelled"}:
            return status
        time.sleep(poll_secs)

def az_register_model_from_job(job_id: str,
                               model_name: str = "iforest-gas",
                               model_relpath: str = "outputs/model_iforest.pkl",
                               model_type: str = "custom_model") -> Optional[str]:
    model_uri = f"azureml://jobs/{job_id}/outputs/artifacts/paths/{model_relpath}"
    rc, out, err = _az("ml", "model", "create", "--name", model_name, "--type", model_type, "--path", model_uri, "-o", "json")
    if rc != 0:
        st.error(f"Error registrando modelo:\nURI: {model_uri}\nSTDERR:\n{err}\nSTDOUT:\n{out}")
        return None
    return str(json.loads(out).get("version"))

def az_endpoint_exists(endpoint_name: str) -> bool:
    rc, _, _ = _az("ml", "online-endpoint", "show", "-n", endpoint_name, "-o", "none")
    return rc == 0

def az_create_endpoint_if_needed(endpoint_name: str, auth_mode: str = "key") -> bool:
    if az_endpoint_exists(endpoint_name):
        return True
    rc, out, err = _az("ml", "online-endpoint", "create", "-n", endpoint_name, "--auth-mode", auth_mode, "-o", "json")
    if rc != 0:
        st.error(f"No pude crear el endpoint '{endpoint_name}'.\nSTDERR:\n{err}\nSTDOUT:\n{out}")
        return False
    return True

def az_deployment_exists(endpoint_name: str, deployment_name: str) -> bool:
    rc, _, _ = _az("ml", "online-deployment", "show", "--name", deployment_name, "--endpoint-name", endpoint_name, "-o", "none")
    return rc == 0

def _pick_existing_env_from_endpoint(endpoint_name: str) -> Optional[str]:
    rc, out, err = _az("ml", "online-deployment", "list", "--endpoint-name", endpoint_name, "-o", "json")
    if rc != 0:
        return None
    items = json.loads(out) or []
    if not items:
        return None
    name = items[0].get("name")
    if not name:
        return None
    rc, out, err = _az("ml", "online-deployment", "show", "--name", name, "--endpoint-name", endpoint_name, "-o", "json")
    if rc != 0:
        return None
    return json.loads(out).get("environment")

def _write_temp_deployment_yaml(endpoint_name: str,
                                deployment_name: str,
                                model_ref: str,
                                instance_type: str,
                                instance_count: int,
                                environment: Optional[str]) -> str:
    lines = [
        "$schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json",
        f"name: {deployment_name}",
        f"endpoint_name: {endpoint_name}",
        f"model: {model_ref}",
        f"instance_type: {instance_type}",
        f"instance_count: {int(instance_count)}",
    ]
    if environment:
        lines.append(f"environment: {environment}")
    content = "\n".join(lines) + "\n"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".yml", mode="w", encoding="utf-8")
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name

def az_update_deployment_model(endpoint_name: str,
                               deployment_name: str,
                               model_name: str,
                               model_version: str,
                               instance_type: str = "Standard_DS3_v2",
                               instance_count: int = 1,
                               environment: Optional[str] = None) -> bool:
    if not az_create_endpoint_if_needed(endpoint_name):
        return False
    model_ref = f"azureml:{model_name}:{model_version}"
    if az_deployment_exists(endpoint_name, deployment_name):
        rc, out, err = _az("ml", "online-deployment", "update",
                           "--name", deployment_name, "--endpoint-name", endpoint_name,
                           "--set", f"model={model_ref}", "-o", "json")
        if rc != 0:
            st.error(f"Error actualizando deployment:\nSTDERR:\n{err}\nSTDOUT:\n{out}")
            return False
        return True
    if not environment:
        environment = _pick_existing_env_from_endpoint(endpoint_name)
        if not environment:
            st.error("No hay deployments previos para clonar el 'environment' y no se proporcionó uno.")
            return False
    yml = _write_temp_deployment_yaml(endpoint_name, deployment_name, model_ref, instance_type, instance_count, environment)
    rc, out, err = _az("ml", "online-deployment", "create", "-f", yml, "-o", "json")
    try:
        os.unlink(yml)
    except Exception:
        pass
    if rc != 0:
        st.error(f"Error creando deployment:\nSTDERR:\n{err}\nSTDOUT:\n{out}")
        return False
    _az("ml", "online-endpoint", "update", "-n", endpoint_name, "--traffic", f"{deployment_name}=100", "-o", "json")
    return True

# =========================
# Transporte HTTP (Requests) con TLS 1.2 opcional
# =========================
class TLS12HttpAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)
    def proxy_manager_for(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        kwargs["ssl_context"] = ctx
        return super().proxy_manager_for(*args, **kwargs)

def _build_session(disable_proxies: bool, force_tls12: bool) -> requests.Session:
    s = requests.Session()
    s.trust_env = not disable_proxies
    if disable_proxies:
        s.proxies = {}
    adapter = TLS12HttpAdapter() if force_tls12 else HTTPAdapter(pool_connections=2, pool_maxsize=2, max_retries=0)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# =========================
# Helpers de respuesta robustos
# =========================
def _as_dict(obj: Any) -> dict:
    """Convierte obj a dict seguro para hacer .get(...)."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        try:
            return json.loads(obj.decode("utf-8"))
        except Exception:
            return {"raw": obj.decode("utf-8", "ignore")}
    if isinstance(obj, str):
        s = obj.strip()
        # intenta JSON
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        except Exception:
            return {"raw": s}
    if isinstance(obj, list):
        return {"result": obj}
    # último recurso
    try:
        return dict(obj)  # puede lanzar
    except Exception:
        return {"raw": repr(obj)}

# =========================
# Preparar payload
# =========================
def _prep_payload(df: pd.DataFrame) -> dict:
    df2 = normalize_headers(df.copy())
    if DATE_COL_DEFAULT in df2.columns:
        df2[DATE_COL_DEFAULT] = pd.to_datetime(df2[DATE_COL_DEFAULT], errors="coerce")
    for col in df2.columns:
        if pd.api.types.is_datetime64_any_dtype(df2[col]):
            df2[col] = df2[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            if df2[col].map(lambda x: isinstance(x, (pd.Timestamp, datetime))).any():
                sdt = pd.to_datetime(df2[col], errors="coerce")
                df2[col] = sdt.dt.strftime("%Y-%m-%d %H:%M:%S")
    if DATE_COL_DEFAULT in df2.columns:
        df2 = df2.dropna(subset=[DATE_COL_DEFAULT]).reset_index(drop=True)
    df2.replace([np.inf, -np.inf], pd.NA, inplace=True)
    df2 = df2.astype(object).where(pd.notnull(df2), None)
    return {"data": df2.to_dict(orient="records")}

# =========================
# Invocaciones (3 modos)
# =========================
def call_via_requests(payload: dict, *, verify_ssl=True, ca_bundle=None, disable_proxies=False, force_tls12=True) -> Tuple[dict, Optional[str]]:
    url = _normalize_url(AML_URL)
    if not url:
        raise RuntimeError("AML_URL no está definido.")
    headers = {"Content-Type": "application/json"}
    if AML_KEY:
        headers["Authorization"] = f"Bearer {AML_KEY}"
    verify_param = True
    if ca_bundle and os.path.exists(ca_bundle):
        verify_param = ca_bundle
    elif not verify_ssl:
        verify_param = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    sess = _build_session(disable_proxies=disable_proxies, force_tls12=force_tls12)
    r = sess.post(url, headers=headers, json=payload, timeout=120, verify=verify_param)
    r.raise_for_status()
    # robusto: intenta JSON, si no, guarda texto
    try:
        out = r.json()
    except ValueError:
        out = {"raw": r.text}
    out = _unwrap_response(out)  # <-- FIX aplicado
    csv_text = None
    if isinstance(out, dict) and out.get("ok") and "csv_base64" in out:
        csv_text = base64.b64decode(out["csv_base64"]).decode("utf-8", errors="ignore")
    return out, csv_text

def call_via_cli_rest(payload: dict) -> Tuple[dict, Optional[str]]:
    """Fallback: llama al scoring_uri con az rest."""
    url = _normalize_url(AML_URL)
    if not url:
        raise RuntimeError("AML_URL no está definido.")
    if not AML_KEY:
        raise RuntimeError("AML_KEY no está definido para invocar con az rest.")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp.flush()
        tmp_path = tmp.name
    rc, out, err = _az(
        "rest", "--method", "post",
        "--uri", url,
        "--headers", f"Authorization=Bearer {AML_KEY}",
        "--headers", "Content-Type=application/json",
        "--body", f"@{tmp_path}",
        "-o", "json"
    )
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    if rc != 0:
        raise RuntimeError(f"Fallo az rest:\nSTDERR:\n{err}\nSTDOUT:\n{out}")
    # robusto: intenta JSON
    try:
        out_json = json.loads(out or "{}")
    except Exception:
        out_json = {"raw": out}
    if not isinstance(out_json, dict):
        out_json = {"result": out_json}
    out_json = _unwrap_response(out_json)  # <-- FIX aplicado
    csv_text = None
    if out_json.get("ok") and "csv_base64" in out_json:
        csv_text = base64.b64decode(out_json["csv_base64"]).decode("utf-8", errors="ignore")
    return out_json, csv_text

def call_via_cli_invoke(payload: dict, *, endpoint_name: str, deployment_name: str,
                        subscription: str = "", resource_group: str = "", workspace: str = "") -> Tuple[dict, Optional[str]]:
    """Recomendado: 'az ml online-endpoint invoke' (plano de control)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp.flush()
        tmp_path = tmp.name

    cmd = [
        "ml", "online-endpoint", "invoke",
        "--name", endpoint_name,
        "--deployment-name", deployment_name,
        "--request-file", tmp_path,
        "-o", "json",
    ]
    if subscription:
        cmd += ["--subscription", subscription]
    if resource_group:
        cmd += ["--resource-group", resource_group]
    if workspace:
        cmd += ["--workspace-name", workspace]

    rc, out, err = _az(*cmd)

    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    if rc != 0:
        raise RuntimeError(f"Fallo 'az ml online-endpoint invoke':\nSTDERR:\n{err}\nSTDOUT:\n{out}")

    # robusto: intenta JSON y garantiza dict
    try:
        out_json = json.loads(out or "{}")
    except Exception:
        out_json = {"raw": out}
    if not isinstance(out_json, dict):
        out_json = {"result": out_json}
    out_json = _unwrap_response(out_json)  # <-- FIX aplicado

    csv_text = None
    if out_json.get("ok") and "csv_base64" in out_json:
        csv_text = base64.b64decode(out_json["csv_base64"]).decode("utf-8", errors="ignore")
    return out_json, csv_text

# =========================
# UI Streamlit
# =========================
st.set_page_config(page_title="Anomalías Gas - IForest (Azure ML)", layout="centered")
st.title("🔎 Detección de Anomalías Gas — Azure ML (flujo combinado)")

with st.sidebar:
    st.subheader("Endpoint (consumo directo)")
    AML_URL = st.text_input("AML_URL (scoring_uri)", value=_normalize_url(AML_URL), help="https://.../score")
    AML_KEY = st.text_input("AML_KEY (primaryKey)", value=AML_KEY, type="password")

    st.subheader("Contexto Azure ML (para 'az ml ... invoke')")
    endpoint_name = st.text_input("Endpoint name", value=DEFAULT_ENDPOINT_NAME)
    deployment_name = st.text_input("Deployment name", value=DEFAULT_DEPLOYMENT_NAME)
    AZ_SUBSCRIPTION_ID = st.text_input("Subscription ID (opcional)", value=AZ_SUBSCRIPTION_ID)
    AZ_RESOURCE_GROUP  = st.text_input("Resource group (opcional)", value=AZ_RESOURCE_GROUP)
    AZ_ML_WORKSPACE    = st.text_input("Workspace name (opcional)", value=AZ_ML_WORKSPACE)

    st.subheader("Modo de invocación")
    USE_AZ_INVOKE = st.checkbox("Usar 'az ml online-endpoint invoke' (recomendado)", value=True)
    USE_AZ_REST   = st.checkbox("Usar 'az rest' (alternativa)", value=False,
                                help="Llama al scoring_uri directo; puede fallar por TLS local.")
    st.caption("Si ninguno está marcado, se usará Requests (Python).")

    st.subheader("Opciones (solo para Requests/Python)")
    IGNORE_SSL = st.checkbox("Ignorar verificación SSL (solo prueba)", value=False)
    CA_BUNDLE = st.text_input("Ruta CA bundle (opcional)", value=os.getenv("REQUESTS_CA_BUNDLE", ""))
    DISABLE_PROXIES = st.checkbox("Desactivar proxies del entorno", value=True)
    FORCE_TLS12 = st.checkbox("Forzar TLS 1.2", value=True)

    if st.button("Guardar .env"):
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"AML_URL={AML_URL}\nAML_KEY={AML_KEY}\n")
            if endpoint_name: f.write(f"AML_ENDPOINT_NAME={endpoint_name}\n")
            if deployment_name: f.write(f"AML_DEPLOYMENT_NAME={deployment_name}\n")
            if AZ_SUBSCRIPTION_ID: f.write(f"AZ_SUBSCRIPTION_ID={AZ_SUBSCRIPTION_ID}\n")
            if AZ_RESOURCE_GROUP:  f.write(f"AZ_RESOURCE_GROUP={AZ_RESOURCE_GROUP}\n")
            if AZ_ML_WORKSPACE:    f.write(f"AZ_ML_WORKSPACE={AZ_ML_WORKSPACE}\n")
            if CA_BUNDLE:          f.write(f"REQUESTS_CA_BUNDLE={CA_BUNDLE}\n")
        st.success("Guardado .env")

tab1, tab2 = st.tabs(["Entrenar (Job Azure ML)", "Detectar (Online Endpoint)"])

with tab1:
    st.markdown("Entrena un **nuevo modelo** con el dataset subido (se crea un **Data Asset `uri_folder`**) y **actualiza el endpoint** con la última versión.")
    file_train = st.file_uploader("Sube CSV/Excel de la planta", type=["csv", "xlsx", "xls"])

    col_a, col_b, col_c, col_d = st.columns(4)
    date_col = col_a.text_input("Columna fecha", DATE_COL_DEFAULT)
    target_col = col_b.text_input("Columna objetivo (caudal)", TARGET_COL_DEFAULT)
    contamination = col_c.number_input("Contamination", min_value=0.001, max_value=0.5, value=0.01, step=0.001, format="%.3f")
    compute_name = col_d.text_input("Compute (AmlCompute)", "cpu-ds1")

    endpoint_name_train = st.text_input("Endpoint (deploy)", endpoint_name)
    deployment_name_train = st.text_input("Deployment (deploy)", deployment_name)
    model_name = st.text_input("Nombre de modelo registrado", "iforest-gas")
    env_ref = st.text_input("Environment (opcional para CREAR)", "", placeholder="azureml:<ENV_NAME>:<VERSION>")

    if st.button("🚀 Entrenar en Azure y actualizar endpoint", key="btn-train"):
        if file_train is None:
            st.warning("Sube un archivo primero.")
        else:
            suffix = ".xlsx" if file_train.name.lower().endswith((".xlsx", ".xls")) else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_train.read())
                local_path = tmp.name

            st.info("Creando job de entrenamiento en Azure ML…")
            job_id = az_job_train(local_path, date_col, target_col, float(contamination), compute=compute_name)
            if job_id:
                st.success(f"Job creado: {job_id}")
                status = az_job_wait(job_id)
                if status == "Completed":
                    st.success("Entrenamiento completado. Registrando modelo…")
                    version = az_register_model_from_job(job_id, model_name=model_name)
                    if version:
                        st.success(f"Modelo registrado: {model_name}:{version}. Actualizando deployment…")
                        ok = az_update_deployment_model(
                            endpoint_name=endpoint_name_train,
                            deployment_name=deployment_name_train,
                            model_name=model_name,
                            model_version=version,
                            instance_type="Standard_DS3_v2",
                            instance_count=1,
                            environment=(env_ref.strip() or None)
                        )
                        if ok:
                            st.success("Deployment listo. ¡El endpoint ya sirve el nuevo modelo!")
                    else:
                        st.error("No se pudo registrar el modelo.")
                else:
                    rc, out, err = _az("ml", "job", "show", "--name", job_id, "-o", "json")
                    run_url = None
                    if rc == 0:
                        try:
                            info = json.loads(out)
                            run_url = info.get("services", {}).get("Studio", {}).get("endpoint")
                        except Exception:
                            pass
                    st.error(f"Job terminó con estado: {status}")
                    if run_url:
                        st.write(f"Revisa logs en Azure ML Studio: {run_url}")

with tab2:
    st.markdown("Envía datos al **Online Endpoint** y descarga el CSV (separador `;`).")
    file_pred = st.file_uploader("Sube CSV/Excel", type=["csv", "xlsx", "xls"], key="pred_file")
    if st.button("💡 Detectar anomalías", key="btn-score"):
        if file_pred is None:
            st.warning("Sube un archivo primero.")
        else:
            try:
                # Leer archivo (autodetección de separador con fallback)
                if file_pred.name.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(io.BytesIO(file_pred.read()))
                else:
                    try:
                        df = pd.read_csv(io.StringIO(file_pred.getvalue().decode("utf-8")), sep=None, engine="python")
                    except Exception:
                        df = pd.read_csv(io.StringIO(file_pred.getvalue().decode("utf-8")), sep=";")
                df = normalize_headers(df)
                st.dataframe(df.head(20), width='stretch')  # <-- FIX deprecación

                payload = _prep_payload(df)

                # 1) Modo recomendado: CLI invoke (plano de control)
                if st.session_state.get("USE_AZ_INVOKE", None) is None:
                    # guardar flags en session_state para fallback en requests si los usas
                    st.session_state["IGNORE_SSL"] = IGNORE_SSL
                    st.session_state["CA_BUNDLE"] = CA_BUNDLE
                    st.session_state["DISABLE_PROXIES"] = DISABLE_PROXIES
                    st.session_state["FORCE_TLS12"] = FORCE_TLS12

                if USE_AZ_INVOKE:
                    out, csv_text = call_via_cli_invoke(
                        payload,
                        endpoint_name=endpoint_name,
                        deployment_name=deployment_name,
                        subscription=AZ_SUBSCRIPTION_ID.strip(),
                        resource_group=AZ_RESOURCE_GROUP.strip(),
                        workspace=AZ_ML_WORKSPACE.strip()
                    )
                elif USE_AZ_REST:
                    out, csv_text = call_via_cli_rest(payload)
                else:
                    out, csv_text = call_via_requests(
                        payload,
                        verify_ssl=(not IGNORE_SSL),
                        ca_bundle=(CA_BUNDLE.strip() or None),
                        disable_proxies=DISABLE_PROXIES,
                        force_tls12=FORCE_TLS12
                    )

                # ---------- Respuesta robusta ----------
                resp = _as_dict(out)
                resp = _unwrap_response(resp)  # <-- FIX aplicado

                if not resp.get("ok"):
                    # si el servidor devolvió otra estructura, muestra algo útil
                    msg = resp.get("error") or resp.get("raw") or "Respuesta sin 'ok' en el cuerpo."
                    st.error(msg)
                    # muestra también la respuesta cruda si existe
                    if "raw" in resp:
                        st.caption("Respuesta raw:")
                        st.code(resp["raw"][:2000])
                else:
                    st.success(f"OK. Filas procesadas: {resp.get('rows')}")
                    preview = resp.get("preview", [])
                    if isinstance(preview, list) and preview:
                        st.dataframe(pd.DataFrame(preview), width='stretch')  # <-- FIX deprecación
                    if csv_text:
                        st.download_button(
                            label="⬇️ Descargar CSV ( ; )",
                            data=csv_text.encode("utf-8"),
                            file_name="anomalies.csv",
                            mime="text/csv"
                        )
                        st.success("CSV listo para descargar.")
            except Exception as e:
                st.error(str(e))


# =========================
# Visualización (serie completa + solo anomalías)
# Se ejecuta si hay resultados en 'out'/'csv_text' o si existe st.session_state['df_out']
# =========================
def _render_anomaly_visuals(df_out: pd.DataFrame):
    st.subheader("Visualización de anomalías")

    # Selector de interpretación para la columna 'anomaly'
    anom_mode = st.radio(
        "¿Qué valor indica 'Anomalía' en la columna 'anomaly'?",
        options=["auto", "1 (anomalía)", "-1 (anomalía)"],
        horizontal=True,
        index=0,
        help=(
            "auto: si hay -1 y 1, toma -1 como anomalía (convención sklearn). "
            "Si hay 0 y 1, toma 1 como anomalía. También soporta otras combinaciones."
        ),
        key="anom_mode_radio"
    )

    # Eje X
    x_col = "FechaHora" if "FechaHora" in df_out.columns else None
    if x_col:
        try:
            df_out[x_col] = pd.to_datetime(df_out[x_col], errors="coerce")
        except Exception:
            pass

    # Eje Y
    y_col = "VolCorrected" if "VolCorrected" in df_out.columns else ("Vol" if "Vol" in df_out.columns else None)

    # Etiquetas de anomalía
    if "anomaly" not in df_out.columns:
        df_out["AnomalyLabel"] = "Desconocido"
    else:
        vals = pd.Series(df_out["anomaly"]).dropna().unique().tolist()

        def is_anom(v):
            try:
                v = int(v)
            except Exception:
                return False
            if anom_mode.startswith("1"):
                return v == 1
            if anom_mode.startswith("-1"):
                return v == -1
            # auto
            if (-1 in vals) and (1 in vals):
                return v == -1
            if (0 in vals) and (1 in vals):
                return v == 1
            return v not in (0, 1)

        df_out["AnomalyLabel"] = df_out["anomaly"].map(lambda v: "Anomalía" if is_anom(v) else "Normal")

    if y_col is None:
        st.info("No encuentro columnas numéricas para el eje Y ('VolCorrected' o 'Vol').")
        return

    if x_col is None:
        df_out = df_out.reset_index(drop=False).rename(columns={"index": "idx"})
        x_col = "idx"

    df_norm = df_out[df_out["AnomalyLabel"] != "Anomalía"]
    df_anom = df_out[df_out["AnomalyLabel"] == "Anomalía"]

    # Graf 1: serie completa
    fig_all = go.Figure()
    if not df_norm.empty:
        fig_all.add_trace(go.Scatter(
            x=df_norm[x_col], y=df_norm[y_col],
            mode="markers", name="Normal",
            hovertemplate=f"{x_col}: "+"%{x}<br>"+f"{y_col}: "+"%{y}<extra></extra>",
            marker=dict(size=6, opacity=0.45)
        ))
    if not df_anom.empty:
        fig_all.add_trace(go.Scatter(
            x=df_anom[x_col], y=df_anom[y_col],
            mode="markers", name="Anomalía",
            hovertemplate=f"{x_col}: "+"%{x}<br>"+f"{y_col}: "+"%{y}<extra></extra>",
            marker=dict(size=11, symbol="x", line=dict(width=2))
        ))
    fig_all.update_layout(
        title="Serie completa con anomalías resaltadas",
        xaxis_title=str(x_col), yaxis_title=str(y_col),
        legend_title_text="Clase", margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # Graf 2: solo anomalías
    if df_anom.empty:
        st.info("No se detectaron anomalías para la vista de zoom.")
    else:
        fig_anom = px.scatter(
            df_anom, x=x_col, y=y_col,
            title="Solo anomalías (zoom)",
            hover_data=[c for c in ["FechaHora", "Vol", "VolCorrected", "anomaly"] if c in df_anom.columns]
        )
        fig_anom.update_traces(marker=dict(size=11, symbol="x", line=dict(width=2)))
        fig_anom.update_layout(
            legend_title_text="Anomalías",
            xaxis_title=str(x_col), yaxis_title=str(y_col),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_anom, use_container_width=True)

    counts = df_out["AnomalyLabel"].value_counts(dropna=False)
    st.caption("Resumen — " + ", ".join([f"{k}: {v}" for k, v in counts.items()]))

# Intentar obtener df_out del resultado reciente o de sesión
try:
    _df_vis = None
    # Si la invocación dejó 'out' y/o 'csv_text' en el scope, intentar reconstruir
    if 'out' in locals() and isinstance(locals().get('out'), dict):
        if isinstance(out.get('preview'), list) and out['preview']:
            _df_vis = pd.DataFrame(out['preview'])
    if _df_vis is None and 'csv_text' in locals() and locals().get('csv_text'):
        _csv = locals().get('csv_text')
        # separador flexible ; o ,
        import re
        sep = ";" if _csv.count(";") >= _csv.count(",") else ","
        _df_vis = pd.read_csv(StringIO(_csv), sep=sep)

    # Guardar en sesión si lo tenemos
    if _df_vis is not None and not _df_vis.empty:
        st.session_state['df_out'] = _df_vis.copy()

    # Usar lo de sesión si existe
    if _df_vis is None and 'df_out' in st.session_state:
        _df_vis = st.session_state['df_out']

    if _df_vis is not None and isinstance(_df_vis, pd.DataFrame) and not _df_vis.empty:
        st.divider()
        _render_anomaly_visuals(_df_vis)
except Exception as _e:
    # No interrumpir el flujo principal si la visual fallara
    st.caption(f"(visualización omitida: {_e})")








