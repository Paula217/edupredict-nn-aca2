"""Carga, descarga y limpieza determinista del dataset UCI 697.

La limpieza de este módulo es **determinista y no aprendida** (no depende de
estadísticas del conjunto de datos), por lo que puede aplicarse antes de la
división entrenamiento/prueba sin generar fuga de información:

1. Limpieza de nombres de columnas (BOM UTF-8, tabuladores y espacios).
2. Renombrado a identificadores internos ``snake_case``.
3. Conversión de tipos (enteros para códigos y conteos, flotantes para notas).
4. Verificación de nulos, duplicados, rangos válidos y consistencia lógica.

No se eliminan valores extremos: las edades altas (p. ej. 70 años) y las notas
de 0 corresponden a situaciones reales (estudiantes mayores de 23 años que
ingresan por cupo especial; semestres sin asignaturas aprobadas). Eliminar
estos registros sesgaría precisamente al grupo con mayor deserción.
"""

from __future__ import annotations

import io
import re
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import data_dictionary as dd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "data.csv"
CLEAN_PATH = DATA_DIR / "estudiantes_limpio.csv"

UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
EXPECTED_ROWS = 4424
EXPECTED_FEATURES = 36


class DataValidationError(ValueError):
    """Error de esquema o consistencia en los datos."""


# ---------------------------------------------------------------------------
# Descarga / lectura
# ---------------------------------------------------------------------------
def download_raw(dest: Path = RAW_PATH, url: str = UCI_ZIP_URL, timeout: int = 60) -> Path:
    """Descarga el ZIP oficial de UCI y extrae ``data.csv`` en ``dest``."""
    logger.info("Descargando dataset desde %s", url)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (URL fija)
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
    logger.info("Dataset guardado en %s", dest)
    return dest


def load_raw(path: Path = RAW_PATH, download_if_missing: bool = False) -> pd.DataFrame:
    """Lee el CSV original de UCI (separador ``;``, codificación UTF-8 con BOM)."""
    path = Path(path)
    if not path.exists():
        if download_if_missing:
            download_raw(path)
        else:
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecute `python -m src.train --download` "
                "o copie el CSV de UCI en data/data.csv."
            )
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def normalizar_nombre(col: str) -> str:
    """Normalización de nombres del notebook borrador.

    ``"Mother's qualification" -> "mothers_qualification"``. Además se corrige la
    errata ``Nacionality -> nationality``. ``COLUMN_MAP`` (data_dictionary) es el
    resultado de aplicar esta función a las columnas de UCI; una prueba verifica
    que ambos coincidan.
    """
    col = str(col).replace("\ufeff", "").strip().lower().replace("'", "")
    col = re.sub(r"[^a-z0-9]+", "_", col).strip("_")
    return "nationality" if col == "nacionality" else col


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Quita BOM, tabuladores y espacios sobrantes; renombra a snake_case."""
    out = df.copy()
    out.columns = [str(c).replace("﻿", "").replace("\t", "").strip() for c in out.columns]
    missing = [c for c in dd.COLUMN_MAP if c not in out.columns]
    if missing:
        raise DataValidationError(f"Faltan columnas esperadas en el CSV: {missing}")
    return out.rename(columns=dd.COLUMN_MAP)[list(dd.COLUMN_MAP.values())]


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte códigos y conteos a enteros y notas/indicadores a flotantes."""
    out = df.copy()
    int_cols = dd.NOMINAL_FEATURES + dd.BINARY_FEATURES + [
        c
        for c in dd.NUMERIC_FEATURES
        if c not in {"previous_qualification_grade", "admission_grade", "curricular_units_1st_sem_grade",
                     "curricular_units_2nd_sem_grade", "unemployment_rate", "inflation_rate", "gdp"}
    ]
    for c in int_cols:
        out[c] = pd.to_numeric(out[c], errors="raise").astype("int64")
    for c in ["previous_qualification_grade", "admission_grade", "curricular_units_1st_sem_grade",
              "curricular_units_2nd_sem_grade", "unemployment_rate", "inflation_rate", "gdp"]:
        out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
    out[dd.TARGET] = out[dd.TARGET].astype(str).str.strip()
    return out


def consistency_report(df: pd.DataFrame) -> dict:
    """Resume nulos, duplicados, rangos y reglas lógicas (no modifica datos)."""
    report: dict = {
        "filas": int(len(df)),
        "columnas_predictoras": int(len([c for c in df.columns if c != dd.TARGET])),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "fuera_de_rango": {},
        "reglas": {},
    }
    for col, (lo, hi) in dd.VALID_RANGES.items():
        if col in df:
            n = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                report["fuera_de_rango"][col] = n
    for sem in ("1st", "2nd"):
        enr, app = df[f"curricular_units_{sem}_sem_enrolled"], df[f"curricular_units_{sem}_sem_approved"]
        grade = df[f"curricular_units_{sem}_sem_grade"]
        report["reglas"][f"aprobadas_mayor_inscritas_{sem}"] = int((app > enr).sum())
        report["reglas"][f"nota_positiva_sin_aprobadas_{sem}"] = int(((app == 0) & (grade > 0)).sum())
        report["reglas"][f"sin_inscripcion_{sem}"] = int((enr == 0).sum())
    report["clases_no_validas"] = int((~df[dd.TARGET].isin(dd.CLASSES)).sum())
    report["casos_atipicos_marcados"] = int(soft_rule_flags(df).sum())
    return report


def soft_rule_flags(df: pd.DataFrame) -> pd.Series:
    """Reglas "blandas" del notebook borrador: casos atípicos pero posibles.

    No se eliminan (p. ej. un desertor con 0 materias inscritas es un caso real);
    solo se marcan para auditoría. Devuelve una Serie booleana por fila.
    """
    s1, s2 = "curricular_units_1st_sem_", "curricular_units_2nd_sem_"
    rules = [
        df[s1 + "approved"] > df[s1 + "enrolled"],
        df[s2 + "approved"] > df[s2 + "enrolled"],
        (df[s1 + "grade"] > 0) & (df[s1 + "enrolled"] == 0),
        (df[s2 + "grade"] > 0) & (df[s2 + "enrolled"] == 0),
        (df[s1 + "enrolled"] > 0) & (df[s1 + "grade"] == 0),
        (df[s2 + "enrolled"] > 0) & (df[s2 + "grade"] == 0),
        (df[s1 + "enrolled"] == 0) & (df[s2 + "enrolled"] == 0),
    ]
    return pd.concat(rules, axis=1).any(axis=1)


def clean_dataset(df_raw: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Aplica la limpieza determinista completa y valida el resultado.

    * Elimina duplicados exactos (el dataset oficial no tiene, pero la regla se
      deja explícita para versiones futuras del CSV).
    * Con ``strict=True`` lanza ``DataValidationError`` ante nulos, clases
      desconocidas o valores fuera de los rangos documentados por UCI.
    """
    df = cast_types(clean_column_names(df_raw))
    n_dup = int(df.duplicated().sum())
    if n_dup:
        logger.warning("Se eliminaron %d filas duplicadas exactas", n_dup)
        df = df.drop_duplicates().reset_index(drop=True)
    rep = consistency_report(df)
    if strict:
        problems = []
        if rep["nulos_totales"]:
            problems.append(f"{rep['nulos_totales']} valores nulos")
        if rep["clases_no_validas"]:
            problems.append(f"{rep['clases_no_validas']} filas con clase no válida")
        if rep["fuera_de_rango"]:
            problems.append(f"valores fuera de rango: {rep['fuera_de_rango']}")
        if any(rep["reglas"][k] for k in rep["reglas"] if k.startswith("aprobadas_mayor")):
            problems.append("hay más unidades aprobadas que inscritas")
        if problems:
            raise DataValidationError("; ".join(problems))
    return df


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Carga ``estudiantes_limpio.csv``; si no existe lo regenera desde el CSV original."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    df = clean_dataset(load_raw())
    save_clean(df, path)
    return df


def save_clean(df: pd.DataFrame, path: Path = CLEAN_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa predictores y variable objetivo."""
    return df[dd.ALL_FEATURES].copy(), df[dd.TARGET].copy()


def decode_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia con códigos traducidos a texto en español (solo para gráficos)."""
    out = df.copy()
    for col in dd.NOMINAL_FEATURES + dd.BINARY_FEATURES:
        if col in out:
            out[col] = out[col].map(lambda v, c=col: dd.code_label(c, v))
    if dd.TARGET in out:
        out["clase"] = out[dd.TARGET].map(dd.CLASS_LABELS_ES)
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raw = load_raw(download_if_missing=True)
    clean = clean_dataset(raw)
    save_clean(clean)
    print(consistency_report(clean))
    print(f"Guardado {CLEAN_PATH} con forma {clean.shape}")
