"""Gestión del dataset en SQLite (Paso 1 del ACA 2: base de datos SQL).

SQLite viene incluido con Python (``import sqlite3``): no requiere cuenta,
servidor, pago ni instalación. La base es un único archivo local
(``data/estudiantes.db``) que se versiona en el repositorio, de modo que la
aplicación Streamlit lee los datos desde SQL y no desde un CSV suelto.

Tablas
------
``estudiantes_raw``   CSV original de UCI tal cual (nombres originales).
``estudiantes``       Datos limpios (nombres normalizados, tipos validados).
``resumen_target``    Vista con conteo y porcentaje por clase.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from src.data_loader import DATA_DIR, RAW_PATH, clean_dataset, load_raw

DB_PATH = DATA_DIR / "estudiantes.db"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)


def build_db(path: Path = DB_PATH, raw_csv: Path = RAW_PATH) -> Path:
    """Crea (o recrea) la base SQLite a partir del CSV original."""
    raw = load_raw(raw_csv)
    clean = clean_dataset(raw, strict=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(connect(path)) as con:
        raw.to_sql("estudiantes_raw", con, if_exists="replace", index=False)
        clean.to_sql("estudiantes", con, if_exists="replace", index_label="id_estudiante")
        con.execute("DROP VIEW IF EXISTS resumen_target")
        con.execute(
            "CREATE VIEW resumen_target AS "
            "SELECT target, COUNT(*) AS estudiantes, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM estudiantes), 2) AS porcentaje "
            "FROM estudiantes GROUP BY target ORDER BY estudiantes DESC"
        )
        con.commit()
    return path


def query(sql: str, params: tuple = (), path: Path = DB_PATH) -> pd.DataFrame:
    """Ejecuta una consulta SELECT y devuelve un DataFrame."""
    with closing(connect(path)) as con:
        return pd.read_sql_query(sql, con, params=params)


def load_clean_from_db(path: Path = DB_PATH) -> pd.DataFrame:
    """Lee la tabla ``estudiantes`` (sin la columna id) con los mismos tipos que ``data_loader.load_clean``."""
    df = query("SELECT * FROM estudiantes ORDER BY id_estudiante", path=path).drop(columns="id_estudiante")
    return df


def ensure_db(path: Path = DB_PATH) -> Path:
    """Devuelve la ruta de la base; si no existe, la construye."""
    return Path(path) if Path(path).exists() else build_db(path)


if __name__ == "__main__":
    p = build_db()
    print(f"Base creada: {p} ({p.stat().st_size / 1024:.0f} KB)")
    print(query("SELECT * FROM resumen_target"))
