# score.py
import os
import io
import json
import base64
from typing import Any, Dict, Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Ruta del artefacto del modelo (ajústala si tu deployment usa otra)
MODEL_DIR = os.getenv("AZUREML_MODEL_DIR", "/var/azureml-app/azureml-models/iforest-gas")
MODEL_PATH = os.path.join(MODEL_DIR, "model_iforest.pkl")

# Convenciones de columnas
DATE_COL   = "FechaHora"
TARGET_COL = "VolCorrected"  # señal cruda

# Aliases comunes de fecha
ALIASES = {
    "Hourly_Date": DATE_COL,
    "Fecha_Hora":  DATE_COL,
    "Date":        DATE_COL,
    "date":        DATE_COL,
    "datetime":    DATE_COL,
}

def init():
    """Se ejecuta una vez al iniciar el contenedor."""
    global model
    model = joblib.load(MODEL_PATH)

def _parse_body(raw_body: Any) -> Dict[str, Any]:
    if isinstance(raw_body, (bytes, bytearray)):
        return json.loads(raw_body.decode("utf-8"))
    if isinstance(raw_body, str):
        return json.loads(raw_body)
    if isinstance(raw_body, dict):
        return raw_body
    return json.loads(raw_body.read().decode("utf-8"))

def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    for src, dst in ALIASES.items():
        if src in df2.columns and dst not in df2.columns:
            df2.rename(columns={src: dst}, inplace=True)
    return df2

def _to_numeric_except_date(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if c != DATE_COL:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2

# --------- utilidades para descubrir/forzar columnas esperadas ---------
def _get_expected_feature_names(est) -> Optional[List[str]]:
    """
    Busca los nombres exactos esperados por el modelo/pipeline:
      - feature_names_in_
      - último paso de Pipeline (o cualquiera)
      - ColumnTransformer.get_feature_names_out()
      - named_steps recursivo
    """
    names = getattr(est, "feature_names_in_", None)
    if names is not None:
        return list(names)

    if isinstance(est, Pipeline):
        last_step = est.steps[-1][1]
        names = getattr(last_step, "feature_names_in_", None)
        if names is not None:
            return list(names)
        for _, step in est.steps:
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                return list(names)

    if isinstance(est, ColumnTransformer):
        try:
            out = est.get_feature_names_out()
            if out is not None and len(out) > 0:
                return list(out)
        except Exception:
            pass
        try:
            cols = []
            for _, __, cols_sel in est.transformers_:
                if isinstance(cols_sel, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend(list(cols_sel))
            if cols:
                return list(cols)
        except Exception:
            pass

    try:
        out = est.get_feature_names_out()
        if out is not None and len(out) > 0:
            return list(out)
    except Exception:
        pass

    if hasattr(est, "named_steps"):
        for step in est.named_steps.values():
            names = _get_expected_feature_names(step)
            if names:
                return list(names)

    return None

def _expand_to_expected(X: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    """
    Construye un DataFrame con columnas EXACTAMENTE 'expected'.
    Regla por cada 'e':
      - si X tiene 'e', usarla
      - si no, usar base = e.split('__')[-1] si existe
      - si no, rellenar con 0.0
    """
    out = {}
    for e in expected:
        if e in X.columns:
            out[e] = X[e]
        else:
            base = e.split('__')[-1]
            out[e] = X[base] if base in X.columns else 0.0
    return pd.DataFrame(out, index=X.index)

def _parse_feature_mismatch(err_text: str) -> Tuple[List[str], List[str]]:
    """
    Parsea el mensaje de scikit-learn:
      - 'Feature names unseen at fit time:' -> lista de 'unseen' en el input
      - 'Feature names seen at fit time, yet now missing:' -> lista de 'missing' que espera el modelo
    Devuelve (unseen, missing)
    """
    unseen, missing = [], []
    lines = [l.strip() for l in err_text.splitlines() if l.strip()]
    mode = None
    for ln in lines:
        low = ln.lower()
        if "feature names unseen at fit time" in low:
            mode = "unseen"; continue
        if "feature names seen at fit time" in low and "missing" in low:
            mode = "missing"; continue
        if ln.startswith("- "):
            name = ln[2:].strip()
            # corta un posible sufijo "..." (cuando el mensaje se trunca)
            if name.endswith("..."):
                name = name[:-3]
            if name:
                if mode == "unseen":
                    unseen.append(name)
                elif mode == "missing":
                    missing.append(name)
    return unseen, missing

def _repair_X_from_error(X: pd.DataFrame, err_text: str) -> pd.DataFrame:
    """
    Repara X usando el texto de error de scikit-learn.
      - elimina 'unseen'
      - crea todas las 'missing' (buscando por base e insertando 0.0 si no hay)
    """
    unseen, missing = _parse_feature_mismatch(err_text)
    if unseen:
        X = X.drop(columns=[c for c in unseen if c in X.columns], errors="ignore")
    if missing:
        # asegúrate de tener VolCorrected por si hay prefijos num__VolCorrected
        if TARGET_COL not in X.columns:
            X[TARGET_COL] = 0.0
        for m in missing:
            if m in X.columns:
                continue
            base = m.split("__")[-1]
            if base in X.columns:
                X[m] = X[base]
            else:
                X[m] = 0.0
    return X

def _safe_to_csv(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False, sep=";")
    return buf.getvalue()

def run(raw_body: Any) -> Dict[str, Any]:
    try:
        body = _parse_body(raw_body)
        data = body.get("data")
        if not isinstance(data, list):
            return {"error": "Payload inválido: se esperaba {'data': [...]}."}

        # 1) Entrada y normalización
        df_in = pd.DataFrame(data)
        df_in = _normalize_headers(df_in)

        # 2) ***Garantiza que exista VolCorrected antes de cualquier acceso***
        if TARGET_COL not in df_in.columns:
            df_in[TARGET_COL] = 0.0  # valor neutro

        # 3) Construcción base (numérico excepto fecha)
        X = _to_numeric_except_date(df_in)

        # 4) Features: acepta features listas o genera desde VolCorrected
        has_roll_mean = "roll_mean" in X.columns
        has_roll_std  = "roll_std"  in X.columns

        # aliases antiguos
        if "roll3" in X.columns and not has_roll_mean:
            X["roll_mean"] = X["roll3"]; has_roll_mean = True
        if "roll6" in X.columns and not has_roll_std:
            X["roll_std"]  = X["roll6"];  has_roll_std  = True

        if not (has_roll_mean and has_roll_std):
            s = pd.to_numeric(X[TARGET_COL], errors="coerce")
            if not has_roll_mean:
                X["roll_mean"] = s.rolling(window=3, min_periods=1).mean(); has_roll_mean = True
            if not has_roll_std:
                X["roll_std"]  = s.rolling(window=6, min_periods=1).std().fillna(0.0); has_roll_std  = True

        # 5) Quita solo aliases antiguos
        X = X.drop(columns=[c for c in ("roll3", "roll6") if c in X.columns], errors="ignore")

        # 6) Alinear a lo que espera el modelo (si lo conocemos)
        expected_names = _get_expected_feature_names(model)
        if expected_names:
            X_aligned = _expand_to_expected(X, expected_names)
        else:
            X_aligned = X  # sin info, intentamos directo

        # 7) Predicción con DataFrame; si falla por nombres, reparamos y reintentamos
        try:
            y_pred = model.predict(X_aligned)
        except Exception as e1:
            txt = str(e1)
            if "feature names" in txt.lower():
                # 7.a Reparar desde el propio mensaje de sklearn
                X_fixed = _repair_X_from_error(X_aligned, txt)

                # Si además podemos ver expected_names, fuerce EXACTO:
                if expected_names:
                    X_fixed = _expand_to_expected(X_fixed, expected_names)

                # Reintentar
                y_pred = model.predict(X_fixed)
            else:
                # Si es otro error, intenta ndarray por dimensión como último recurso
                n_expected = getattr(model, "n_features_in_", None)
                num_X = X_aligned.select_dtypes(include=[np.number])
                if n_expected is None:
                    X_arr = num_X.to_numpy(dtype=float)
                else:
                    if num_X.shape[1] < n_expected:
                        pad = np.zeros((num_X.shape[0], n_expected - num_X.shape[1]), dtype=float)
                        X_arr = np.hstack([num_X.to_numpy(dtype=float), pad])
                    else:
                        X_arr = num_X.iloc[:, :n_expected].to_numpy(dtype=float)
                y_pred = model.predict(X_arr)

        # 8) Salida
        out_df = df_in.copy()
        out_df["anomaly"] = (y_pred == -1).astype(int)

        preview = out_df.head(10).to_dict(orient="records")
        csv_text = _safe_to_csv(out_df)
        csv_b64 = base64.b64encode(csv_text.encode("utf-8")).decode("utf-8")

        return {
            "ok": True,
            "rows": int(out_df.shape[0]),
            "preview": preview,
            "csv_base64": csv_b64,
        }

    except Exception as e:
        # Devuelve error claro a la UI/cliente
        return {"error": str(e)}







