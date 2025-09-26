# aml/train.py
import argparse
import os
import glob
import sys
import json
import traceback
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


def log(msg: str):
    print(f"[TRAIN] {msg}", flush=True)


def _debug_fs():
    try:
        for p in ("/", "/mnt", "/mnt/azureml", "/mnt/azureml/inputs"):
            if os.path.exists(p):
                log(f"LS {p}: {os.listdir(p)}")
            else:
                log(f"{p} NO existe")
    except Exception as e:
        log(f"FS debug error: {e}")


def _pick_file_from_dir(d: str) -> str:
    """Elige un archivo dentro de un directorio (prioriza csv/xlsx/xls)."""
    patterns = ("*.csv", "*.xlsx", "*.xls", "*.*")
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(d, pat)))
        if files:
            log(f"Usando archivo de {d}: {files[0]}")
            return files[0]
    raise FileNotFoundError(f"No se encontraron archivos dentro de {d}")


def _resolve_input_path(cli_value: str, input_name: str = "data") -> str:
    """
    Resuelve la ruta real del input 'data' soportando varios casos:
    1) Si cli_value es una ruta válida -> úsala (si es dir, escoge archivo).
    2) Variables de entorno AZUREML_INPUT_<name>.
    3) Montaje típico /mnt/azureml/inputs/<name>.
    4) Descarga local ./inputs/<name> o ./inputs/data.
    5) Como último recurso, primer subdirectorio de /mnt/azureml/inputs.
    """
    _debug_fs()

    # 1) ¿Es ruta válida ya?
    if cli_value and os.path.exists(cli_value):
        if os.path.isdir(cli_value):
            return _pick_file_from_dir(cli_value)
        return cli_value

    # 2) Variables de entorno estándar de AML
    env_key = f"AZUREML_INPUT_{input_name}"
    env_path = os.getenv(env_key)
    if env_path and os.path.exists(env_path):
        log(f"Resuelto por {env_key}: {env_path}")
        if os.path.isdir(env_path):
            return _pick_file_from_dir(env_path)
        return env_path

    # 3) Montaje típico /mnt/azureml/inputs/<name>
    mount_path = f"/mnt/azureml/inputs/{input_name}"
    if os.path.exists(mount_path):
        log(f"Resuelto por mount_path: {mount_path}")
        if os.path.isdir(mount_path):
            return _pick_file_from_dir(mount_path)
        return mount_path

    # 4) Descarga local (download mode)
    for cand in (os.path.join("inputs", input_name),
                 os.path.join("inputs", "data")):
        if os.path.exists(cand):
            log(f"Resuelto por descarga local: {cand}")
            if os.path.isdir(cand):
                return _pick_file_from_dir(cand)
            return cand

    # 5) Primer subdir bajo /mnt/azureml/inputs (fallback)
    inputs_root = "/mnt/azureml/inputs"
    if os.path.exists(inputs_root):
        subdirs = [os.path.join(inputs_root, d) for d in os.listdir(inputs_root)]
        subdirs = [d for d in subdirs if os.path.isdir(d)]
        if subdirs:
            log(f"No se encontró '{input_name}', usando primer subdir: {subdirs[0]}")
            return _pick_file_from_dir(subdirs[0])

    # Nada funcionó
    raise FileNotFoundError(
        f"Ruta de datos no encontrada. Argumento: '{cli_value}'. "
        f"Revisa que el input '{input_name}' sea 'uri_folder' y que el job lo reciba como path real "
        f"(p. ej., usando --set inputs.{input_name}.path=azureml:<asset>@<version>)."
    )


def make_features(df: pd.DataFrame, date_col: str, y_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col, y_col]).sort_values(date_col)

    # Features temporales básicas
    df["hour"] = df[date_col].dt.hour
    df["dow"] = df[date_col].dt.dayofweek

    # Lags / Rolling (tolerante)
    df["lag1"] = df[y_col].shift(1)
    df["roll_mean"] = df[y_col].rolling(24, min_periods=12).mean()
    df["roll_std"] = df[y_col].rolling(24, min_periods=12).std()

    df = df.dropna().reset_index(drop=True)
    return df


def _read_dataframe(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        log(f"Leyendo Excel: {path}")
        return pd.read_excel(path)
    else:
        log(f"Leyendo CSV: {path}")
        try:
            return pd.read_csv(path, sep=None, engine="python")
        except Exception as e1:
            log(f"read_csv auto falló: {e1}; reintento con sep=';'")
            try:
                return pd.read_csv(path, sep=";")
            except Exception as e2:
                log(f"read_csv con ';' falló: {e2}")
                raise


# --- Cambios mínimos para resolver alias de columnas -------------------------

def _norm(s: str) -> str:
    """Normaliza nombre de columna para comparar de forma robusta."""
    return (
        s.strip()
         .lower()
         .replace(" ", "")
         .replace("-", "")
         .replace("/", "")
         .replace("_", "")
    )

def _resolve_column_alias(required_name: str, df_cols) -> str | None:
    """
    Si 'required_name' no existe en df, intenta encontrar un alias razonable.
    Devuelve el nombre real de la columna encontrada o None si no hay match.
    """
    # Mapa de alias conocidos (puedes ampliar si lo necesitas)
    aliases = {
        "fechahora": {
            "fechahora", "fecha_hora", "fecha", "hora", "datetime",
            "timestamp", "date", "hourlydate", "hourly_date", "fechahorA"
        },
        "volcorrected": {
            "volcorrected", "vol_corrected", "volumen_corregido",
            "volumenajustado", "volcorr", "volcorrigido"
        },
    }

    norm_to_real = {_norm(c): c for c in df_cols}
    req_norm = _norm(required_name)

    # 1) coincidencia exacta tras normalizar
    if req_norm in norm_to_real:
        return norm_to_real[req_norm]

    # 2) prueba con conjunto de alias predefinidos
    cand_set = aliases.get(req_norm, set())
    for cand in cand_set:
        if cand in norm_to_real:
            return norm_to_real[cand]

    # 3) heurística: buscar por prefijos/sufijos comunes
    for k, real in norm_to_real.items():
        if k.endswith(req_norm) or req_norm in k or k.startswith(req_norm):
            return real

    return None
# -----------------------------------------------------------------------------


def main(args):
    os.makedirs("outputs", exist_ok=True)

    try:
        data_path = _resolve_input_path(args.data, "data")
        log(f"Ruta de datos final: {data_path}")

        raw = _read_dataframe(data_path)
        log(f"Columnas detectadas: {list(raw.columns)}")

        # --- NUEVO: remapeo ligero de alias si falta la columna esperada -----
        # date_col
        if args.date_col not in raw.columns:
            resolved = _resolve_column_alias(args.date_col, raw.columns)
            if resolved:
                log(f"date_col '{args.date_col}' no encontrado. Usando alias '{resolved}'.")
                args.date_col = resolved
        # target_col
        if args.target_col not in raw.columns:
            resolved = _resolve_column_alias(args.target_col, raw.columns)
            if resolved:
                log(f"target_col '{args.target_col}' no encontrado. Usando alias '{resolved}'.")
                args.target_col = resolved
        # ---------------------------------------------------------------------

        # Validaciones básicas
        if args.date_col not in raw.columns or args.target_col not in raw.columns:
            raise ValueError(
                f"Columnas no encontradas. date_col='{args.date_col}' "
                f"target_col='{args.target_col}'. Columnas disponibles: {list(raw.columns)}"
            )

        feats = make_features(raw, args.date_col, args.target_col)
        log(f"Filas tras features: {len(feats)}")
        if len(feats) < 24:
            raise RuntimeError(
                "Se requieren al menos 24 filas válidas tras features (lags/rolling)."
            )

        # Construir X con solo numéricas
        drop_cols = [c for c in [args.date_col, args.target_col] if c in feats.columns]
        X = feats.drop(columns=drop_cols, errors="ignore").select_dtypes("number")
        if X.empty:
            raise ValueError("No hay columnas numéricas para entrenar después de features.")

        # Entrenar IsolationForest
        model = IsolationForest(
            n_estimators=300,
            contamination=float(args.contamination),
            random_state=42,
            n_jobs=-1,
        ).fit(X)

        # Guardar artefactos
        joblib.dump(model, "outputs/model_iforest.pkl")
        joblib.dump({"date_col": args.date_col, "y_col": args.target_col}, "outputs/meta.pkl")
        joblib.dump(list(X.columns), "outputs/feature_cols.pkl")

        # Preview opcional
        feats.head(10).to_csv("outputs/train_preview.csv", index=False)

        log("Entrenamiento COMPLETADO. Artefactos guardados en outputs/")
    except Exception as e:
        log("ERROR en entrenamiento")
        log(str(e))
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Ruta a CSV/Excel o directorio montado/descargado.")
    p.add_argument("--date_col", default="FechaHora")
    p.add_argument("--target_col", default="VolCorrected")
    p.add_argument("--contamination", type=float, default=0.01)
    args = p.parse_args()
    main(args)


