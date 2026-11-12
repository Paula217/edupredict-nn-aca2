"""Carga de artefactos y predicción con las redes neuronales entrenadas (ACA 2).

La aplicación Streamlit y las pruebas usan exclusivamente este módulo, de modo
que la entrada del formulario pasa por la **misma** validación de esquema y el
**mismo** codificador + red neuronal que se usaron en entrenamiento.
El modelo se guarda como ``models/modelo_nn*.pt`` (PyTorch) y se carga con
``NNClassifier``, que expone ``predict_proba`` / ``classes_`` como scikit-learn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src import data_dictionary as dd
from src.data_loader import ROOT
from src.preprocessing import FEATURE_SETS, FeatureSet, validate_input

MODELS_DIR = ROOT / "models"

ARTIFACTS = {
    "completo": ("modelo_nn.pt", "metadata_nn.json", "metricas_nn.json"),
    "temprano": ("modelo_nn_temprano.pt", "metadata_nn_temprano.json", "metricas_nn_temprano.json"),
}


class ModelNotAvailableError(FileNotFoundError):
    """No se encontraron los artefactos entrenados."""


@dataclass
class PredictionResult:
    clase: str
    clase_es: str
    probabilidades: dict[str, float]
    advertencias: list[str] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "clase": list(self.probabilidades),
            "clase_es": [dd.CLASS_LABELS_ES[c] for c in self.probabilidades],
            "probabilidad": list(self.probabilidades.values()),
        })


def artifact_paths(modality: str, models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    if modality not in ARTIFACTS:
        raise ValueError(f"Modalidad desconocida: {modality}. Opciones: {list(ARTIFACTS)}")
    m, md, mt = ARTIFACTS[modality]
    return {"modelo": models_dir / m, "metadata": models_dir / md, "metricas": models_dir / mt}


def check_artifacts(models_dir: Path = MODELS_DIR) -> dict[str, list[str]]:
    """Devuelve, por modalidad, la lista de artefactos faltantes (vacía si todo existe)."""
    return {
        mod: [p.name for p in artifact_paths(mod, models_dir).values() if not p.exists()]
        for mod in ARTIFACTS
    }


@lru_cache(maxsize=4)
def _load(modality: str, models_dir: str) -> tuple[object, dict, dict]:
    paths = artifact_paths(modality, Path(models_dir))
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise ModelNotAvailableError(
            f"Faltan artefactos del modelo '{modality}': {missing}. Ejecute el notebook "
            "notebooks/ACA2_Student_Dropout_NN.ipynb (sección de exportación) para generarlos."
        )
    from src.nn.wrapper import NNClassifier  # import diferido: PyTorch solo se carga al predecir

    model = NNClassifier.load(paths["modelo"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    metrics = json.loads(paths["metricas"].read_text(encoding="utf-8"))
    expected = FEATURE_SETS[modality].raw_columns
    if metadata["columnas_entrada"] != expected:
        raise ModelNotAvailableError(
            f"El artefacto '{modality}' fue entrenado con otro esquema de columnas. "
            "Reentrene ejecutando el notebook del ACA 2."
        )
    return model, metadata, metrics


def load_model(modality: str = "completo", models_dir: Path = MODELS_DIR):
    return _load(modality, str(models_dir))[0]


def load_metadata(modality: str = "completo", models_dir: Path = MODELS_DIR) -> dict:
    return _load(modality, str(models_dir))[1]


def load_metrics(modality: str = "completo", models_dir: Path = MODELS_DIR) -> dict:
    return _load(modality, str(models_dir))[2]


def feature_set(modality: str) -> FeatureSet:
    return FEATURE_SETS[modality]


def predict_frame(df: pd.DataFrame, modality: str = "completo",
                  models_dir: Path = MODELS_DIR) -> tuple[pd.DataFrame, list[str]]:
    """Predice un lote. Devuelve DataFrame con clase y probabilidades + advertencias."""
    model = load_model(modality, models_dir)
    X, warnings = validate_input(df, feature_set(modality))
    proba = model.predict_proba(X)
    classes = [str(c) for c in model.classes_]
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=X.index)
    out.insert(0, "prediccion", [classes[i] for i in proba.argmax(axis=1)])
    return out, warnings


def predict_one(record: dict, modality: str = "completo",
                models_dir: Path = MODELS_DIR) -> PredictionResult:
    """Predice para un único estudiante descrito como diccionario columna -> valor."""
    res, warnings = predict_frame(pd.DataFrame([record]), modality, models_dir)
    row = res.iloc[0]
    probs = {c: float(row[f"p_{c}"]) for c in dd.CLASSES if f"p_{c}" in row}
    clase = str(row["prediccion"])
    return PredictionResult(clase=clase, clase_es=dd.CLASS_LABELS_ES[clase],
                            probabilidades=probs, advertencias=warnings)


def describe_prediction(result: PredictionResult) -> str:
    """Interpretación descriptiva (no prescriptiva) del resultado."""
    p = result.probabilidades
    ordered = sorted(p.items(), key=lambda kv: kv[1], reverse=True)
    (c1, p1), (c2, p2) = ordered[0], ordered[1]
    margin = p1 - p2
    if margin < 0.15:
        certeza = (
            f"El modelo no distingue con claridad entre **{dd.CLASS_LABELS_ES[c1]}** "
            f"({p1:.0%}) y **{dd.CLASS_LABELS_ES[c2]}** ({p2:.0%}); el perfil es ambiguo."
        )
    elif p1 >= 0.7:
        certeza = (
            f"El perfil se parece mayoritariamente al de estudiantes que en los datos "
            f"históricos terminaron en **{dd.CLASS_LABELS_ES[c1]}** ({p1:.0%})."
        )
    else:
        certeza = (
            f"La clase más probable es **{dd.CLASS_LABELS_ES[c1]}** ({p1:.0%}), seguida de "
            f"**{dd.CLASS_LABELS_ES[c2]}** ({p2:.0%})."
        )
    extra = ""
    if result.clase != "Dropout" and p.get("Dropout", 0) >= 0.30:
        extra = (
            f" Aunque no es la clase principal, la probabilidad estimada de deserción "
            f"({p['Dropout']:.0%}) no es despreciable y podría justificar seguimiento académico."
        )
    return certeza + extra
