"""Preprocesamiento con pipelines de Scikit-learn y validación del esquema de entrada.

Principios:

* Toda transformación **aprendida** (escalado, codificación con frecuencias
  mínimas) vive dentro de un ``Pipeline`` y se ajusta solo con los pliegues de
  entrenamiento, evitando fuga de información hacia validación o prueba.
* Cada algoritmo recibe el preprocesamiento que le conviene:

  - Regresión logística: One-Hot para nominales + estandarización de numéricas
    (el modelo es sensible a la escala y a la codificación ordinal arbitraria).
  - Random Forest: One-Hot para nominales, numéricas sin escalar (los árboles
    son invariantes a transformaciones monótonas).
  - HistGradientBoosting: codificación ordinal + soporte **nativo** de
    categóricas, numéricas sin escalar.

* Las categorías no vistas en entrenamiento no rompen la predicción: One-Hot
  las envía al grupo "infrecuente" y el codificador ordinal las marca como
  valor faltante, que HistGradientBoosting maneja de forma nativa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import data_dictionary as dd

# Categorías con menos de este número de apariciones en el pliegue de
# entrenamiento se agrupan como "infrecuentes" (reduce dimensionalidad de las
# ocupaciones y niveles educativos de los padres, con 29–46 códigos).
MIN_CATEGORY_FREQUENCY = 10


# ---------------------------------------------------------------------------
# Conjuntos de variables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSet:
    """Define qué columnas usa una modalidad de predicción."""

    name: str
    description: str
    nominal: tuple[str, ...]
    binary: tuple[str, ...]
    numeric: tuple[str, ...]
    semesters: tuple[str, ...]

    @property
    def raw_columns(self) -> list[str]:
        """Columnas que el usuario/pipeline debe suministrar."""
        return list(self.nominal) + list(self.binary) + list(self.numeric)

    @property
    def engineered(self) -> list[str]:
        return [f"curricular_units_{s}_sem_approval_rate" for s in self.semesters]

    @property
    def numeric_with_engineered(self) -> list[str]:
        return list(self.numeric) + self.engineered


FULL = FeatureSet(
    name="completo",
    description=(
        "Información académica completa: datos de ingreso, socioeconómicos, "
        "macroeconómicos y rendimiento del 1.er y 2.º semestre."
    ),
    nominal=tuple(dd.NOMINAL_FEATURES),
    binary=tuple(dd.BINARY_FEATURES),
    numeric=tuple(dd.NUMERIC_FEATURES),
    semesters=("1st", "2nd"),
)

EARLY = FeatureSet(
    name="temprano",
    description=(
        "Alerta temprana al cierre del 1.er semestre: excluye todas las variables "
        "del 2.º semestre (información futura en ese momento)."
    ),
    nominal=tuple(dd.NOMINAL_FEATURES),
    binary=tuple(dd.BINARY_FEATURES),
    numeric=tuple(c for c in dd.NUMERIC_FEATURES if c not in dd.SECOND_SEMESTER_FEATURES),
    semesters=("1st",),
)

FEATURE_SETS = {FULL.name: FULL, EARLY.name: EARLY}


# ---------------------------------------------------------------------------
# Ingeniería de variables (sin estado -> sin riesgo de fuga)
# ---------------------------------------------------------------------------
class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Añade la tasa de aprobación por semestre (aprobadas / inscritas).

    Es una transformación fila a fila, sin parámetros aprendidos. Si el
    estudiante no inscribió unidades, la tasa se define como 0 (no aprobó
    ninguna unidad), lo que es coherente con la nota 0 registrada en UCI.
    """

    def __init__(self, semesters: tuple[str, ...] = ("1st", "2nd")):
        self.semesters = semesters

    def fit(self, X, y=None):  # noqa: D401, N803
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):  # noqa: N803
        X = X.copy()
        for s in self.semesters:
            enr = X[f"curricular_units_{s}_sem_enrolled"].astype(float)
            app = X[f"curricular_units_{s}_sem_approved"].astype(float)
            X[f"curricular_units_{s}_sem_approval_rate"] = np.where(enr > 0, app / enr.replace(0, np.nan), 0.0)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(base + [f"curricular_units_{s}_sem_approval_rate" for s in self.semesters], dtype=object)


class WinsorizadorIQR(BaseEstimator, TransformerMixin):
    """Recorta valores fuera de [Q1 - k·IQR, Q3 + k·IQR] (tomado del notebook borrador).

    Los límites se aprenden en ``fit`` (solo con datos de entrenamiento), por lo
    que puede vivir dentro del pipeline sin fuga de información. En EduPredict
    no se aplica de forma fija: es una **opción** de la rejilla de la regresión
    logística (``winsor ∈ {passthrough, WinsorizadorIQR}``) y la validación
    cruzada decide si aporta. Recortar la edad afecta a ~10 % de los
    estudiantes (todos los mayores de ~34 años), que son casos legítimos.
    """

    def __init__(self, columnas=("previous_qualification_grade", "admission_grade", "age_at_enrollment"),
                 k: float = 1.5):
        self.columnas = columnas
        self.k = k

    def fit(self, X, y=None):  # noqa: N803
        self.limites_ = {}
        for c in self.columnas:
            q1, q3 = X[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.limites_[c] = (float(q1 - self.k * iqr), float(q3 + self.k * iqr))
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        return self

    def transform(self, X):  # noqa: N803
        X = X.copy()
        for c, (lo, hi) in self.limites_.items():
            X[c] = X[c].clip(lo, hi)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


# ---------------------------------------------------------------------------
# ColumnTransformers por tipo de modelo
# ---------------------------------------------------------------------------
def _onehot() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=MIN_CATEGORY_FREQUENCY,
        sparse_output=False,
    )


def build_preprocessor(model_kind: str, fs: FeatureSet) -> ColumnTransformer:
    """Construye el ColumnTransformer adecuado para ``model_kind``.

    ``model_kind`` ∈ {"logreg", "rf", "hgb"}. Las columnas nominales siempre
    quedan **primero** en la salida; ``hgb`` usa esa posición para declarar sus
    variables categóricas nativas.
    """
    nominal, binary, numeric = list(fs.nominal), list(fs.binary), fs.numeric_with_engineered
    if model_kind == "logreg":
        transformers = [
            ("nominal", _onehot(), nominal),
            ("binary", "passthrough", binary),
            ("numeric", StandardScaler(), numeric),
        ]
    elif model_kind == "rf":
        transformers = [
            ("nominal", _onehot(), nominal),
            ("binary", "passthrough", binary),
            ("numeric", "passthrough", numeric),
        ]
    elif model_kind == "hgb":
        ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
            min_frequency=MIN_CATEGORY_FREQUENCY,
            dtype=np.float64,
        )
        transformers = [
            ("nominal", ordinal, nominal),
            ("binary", "passthrough", binary),
            ("numeric", "passthrough", numeric),
        ]
    else:  # pragma: no cover - error de programación
        raise ValueError(f"Tipo de modelo desconocido: {model_kind}")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# Validación del esquema de entrada (interfaz y API de predicción)
# ---------------------------------------------------------------------------
class SchemaError(ValueError):
    """La entrada no cumple el esquema esperado por el modelo."""


def validate_input(df: pd.DataFrame, fs: FeatureSet) -> tuple[pd.DataFrame, list[str]]:
    """Valida y normaliza un DataFrame de entrada para la modalidad ``fs``.

    Devuelve ``(df_normalizado, advertencias)``. Lanza ``SchemaError`` si faltan
    columnas, hay valores no numéricos, nulos, valores fuera de rango o
    inconsistencias lógicas (aprobadas > inscritas). Las columnas sobrantes se
    descartan con una advertencia; los códigos categóricos no vistos generan una
    advertencia (el pipeline los maneja como categoría infrecuente/faltante).
    """
    if not isinstance(df, pd.DataFrame):
        raise SchemaError("La entrada debe ser un pandas.DataFrame")
    cols = fs.raw_columns
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"Faltan columnas requeridas: {missing}")
    warnings: list[str] = []
    extra = [c for c in df.columns if c not in cols]
    if extra:
        warnings.append(f"Se ignoran columnas no usadas por el modelo {fs.name}: {extra}")
    out = df[cols].copy()
    for c in cols:
        try:
            out[c] = pd.to_numeric(out[c], errors="raise")
        except (ValueError, TypeError) as exc:
            raise SchemaError(f"La columna '{c}' debe ser numérica") from exc
    if out.isna().any().any():
        raise SchemaError(f"Hay valores nulos en: {out.columns[out.isna().any()].tolist()}")
    for c in fs.binary:
        if not out[c].isin([0, 1]).all():
            raise SchemaError(f"La columna binaria '{c}' solo admite 0 o 1")
    for c in fs.numeric:
        lo, hi = dd.VALID_RANGES[c]
        if ((out[c] < lo) | (out[c] > hi)).any():
            raise SchemaError(f"'{dd.label(c)}' fuera de rango [{lo}, {hi}]")
    for s in fs.semesters:
        if (out[f"curricular_units_{s}_sem_approved"] > out[f"curricular_units_{s}_sem_enrolled"]).any():
            raise SchemaError(
                f"{dd.label(f'curricular_units_{s}_sem_approved')} no puede superar {dd.label(f'curricular_units_{s}_sem_enrolled')}"
            )
    for c in fs.nominal:
        known = dd.CATEGORY_CODES[c]
        unknown = sorted(set(out[c].astype(int)) - set(known))
        if unknown:
            warnings.append(
                f"Código(s) no documentado(s) en '{dd.label(c)}': {unknown}; "
                "se tratarán como categoría infrecuente."
            )
    int_cols = list(fs.nominal) + list(fs.binary)
    out[int_cols] = out[int_cols].astype("int64")
    return out, warnings
