"""Entrenamiento reproducible de EduPredict AI.

Uso::

    python -m src.train            # usa data/data.csv
    python -m src.train --download # descarga primero el CSV oficial de UCI

Protocolo (idéntico para las dos modalidades, *completo* y *temprano*), basado
en las Partes 3–4 del notebook ``ACA1_Student_Dropout`` (basado en el borrador del equipo):

1. Limpieza determinista (``data_loader.clean_dataset``).
2. División estratificada 80/20 con ``RANDOM_STATE = 42``. El 20 % de prueba
   se aparta y **no** interviene en ninguna decisión de modelado.
3. Para cada algoritmo (regresión logística, random forest,
   HistGradientBoosting): ``GridSearchCV`` con validación cruzada estratificada
   de 5 pliegues sobre el 80 % de entrenamiento, optimizando **macro-F1**. Las
   rejillas son las del borrador, ampliadas con dos decisiones que antes eran
   fijas y ahora se validan empíricamente: la ponderación de clases y (solo en
   la regresión logística) la winsorización IQR.
4. Selección del modelo ganador por macro-F1 medio de validación cruzada
   (corrección respecto al borrador, que elegía con el conjunto de prueba).
5. Evaluación final de los tres modelos en prueba (informativa) y
   exportación del pipeline ganador.

El notebook usa ``fit_modality`` para ejecutar exactamente este mismo
entrenamiento y ``export_modality`` para guardar los artefactos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src import data_dictionary as dd
from src.data_loader import RAW_PATH, ROOT, clean_dataset, consistency_report, load_raw, save_clean
from src.evaluate import compute_metrics, cv_scoring, permutation_importances, summarize_cv
from src.predict import artifact_paths
from src.preprocessing import (
    EARLY, FULL, AcademicFeatureEngineer, FeatureSet, WinsorizadorIQR, build_preprocessor,
)

logger = logging.getLogger("edupredict.train")

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SELECTION_METRIC = "f1_macro"
MODELS_DIR = ROOT / "models"

MODEL_NAMES = {
    "logreg": "Regresión Logística",
    "rf": "Random Forest",
    "hgb": "Gradient Boosting (HistGradientBoosting)",
}


def build_estimator(kind: str, fs: FeatureSet, default_class_weight: str | None = None):
    """Devuelve (pipeline, rejilla de hiperparámetros) para un algoritmo.

    Estructura del pipeline: ``features`` (tasas de aprobación, sin estado) →
    ``winsor`` (``passthrough`` por defecto) → ``prep`` (ColumnTransformer
    específico) → ``clf``. ``default_class_weight`` solo afecta al estimador
    sin ajustar (útil para la CV base del notebook).
    """
    if kind == "logreg":
        clf = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, class_weight=default_class_weight)
        grid = {
            "winsor": ["passthrough", WinsorizadorIQR()],
            "clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        }
    elif kind == "rf":
        cw = "balanced_subsample" if default_class_weight else None
        clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1, class_weight=cw)
        grid = {
            "clf__max_depth": [8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.5],
            "clf__class_weight": [None, "balanced_subsample"],
        }
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(
            early_stopping=False,
            categorical_features=list(range(len(fs.nominal))),  # categóricas nativas
            random_state=RANDOM_STATE,
            class_weight=default_class_weight,
        )
        grid = {
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [3, 6],
            "clf__min_samples_leaf": [20, 40],
            "clf__max_iter": [150, 300],
            "clf__class_weight": [None, "balanced"],
        }
    else:  # pragma: no cover
        raise ValueError(kind)
    pipe = Pipeline([
        ("features", AcademicFeatureEngineer(semesters=fs.semesters)),
        ("winsor", "passthrough"),
        ("prep", build_preprocessor(kind, fs)),
        ("clf", clf),
    ])
    return pipe, grid


def split_data(df: pd.DataFrame, fs: FeatureSet):
    """Partición estratificada 80/20 reproducible (misma para ambas modalidades)."""
    X = df[fs.raw_columns].copy()
    y = df[dd.TARGET].copy()
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def make_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _param_repr(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, WinsorizadorIQR):
        return f"WinsorizadorIQR(k={v.k})"
    return v


@dataclass
class ModalityResult:
    fs: FeatureSet
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    searches: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    winner: str = ""

    @property
    def best_pipeline(self):
        return self.searches[self.winner].best_estimator_


def fit_modality(df: pd.DataFrame, fs: FeatureSet, n_jobs: int = -1) -> ModalityResult:
    """Ejecuta GridSearchCV para los tres algoritmos y selecciona por CV."""
    X_train, X_test, y_train, y_test = split_data(df, fs)
    res = ModalityResult(fs, X_train, X_test, y_train, y_test)
    scoring = cv_scoring()
    for kind in MODEL_NAMES:
        pipe, grid = build_estimator(kind, fs)
        search = GridSearchCV(pipe, grid, scoring=scoring, refit=SELECTION_METRIC, cv=make_cv(),
                              n_jobs=n_jobs, return_train_score=True, error_score="raise")
        t0 = time.perf_counter()
        logger.info("[%s] GridSearchCV %s (%d combinaciones)", fs.name, kind,
                    int(np.prod([len(v) for v in grid.values()])))
        search.fit(X_train, y_train)
        best = search.best_estimator_
        test = compute_metrics(y_test, best.predict(X_test), best.predict_proba(X_test),
                               classes=list(best.classes_))
        res.searches[kind] = search
        res.results[kind] = {
            "nombre": MODEL_NAMES[kind],
            "mejores_hiperparametros": {k.replace("clf__", ""): _param_repr(v)
                                        for k, v in search.best_params_.items()},
            "combinaciones_evaluadas": int(len(search.cv_results_["params"])),
            "cv": summarize_cv(search, list(scoring)),
            "prueba": test,
            "tiempo_busqueda_s": round(time.perf_counter() - t0, 1),
        }
        cv = res.results[kind]["cv"]["f1_macro"]
        logger.info("[%s] %s CV macro-F1=%.4f ± %.4f (train %.4f) | prueba macro-F1=%.4f", fs.name, kind,
                    cv["media"], cv["desv"], cv["media_train"], test["f1_macro"])
    # Selección SOLO con validación cruzada (el conjunto de prueba no interviene).
    res.winner = max(res.results, key=lambda k: res.results[k]["cv"][SELECTION_METRIC]["media"])
    return res


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def export_modality(res: ModalityResult, df: pd.DataFrame, out_dir: Path = MODELS_DIR) -> dict:
    """Guarda el pipeline ganador, sus métricas y metadatos. Devuelve las métricas."""
    fs, winner = res.fs, res.winner
    best_pipe = res.best_pipeline
    importances = permutation_importances(best_pipe, res.X_test, res.y_test, random_state=RANDOM_STATE)
    ranking = sorted(res.results, key=lambda k: res.results[k]["cv"][SELECTION_METRIC]["media"], reverse=True)

    paths = artifact_paths(fs.name, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, paths["modelo"], compress=3)

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metricas = {
        "modalidad": fs.name,
        "descripcion": fs.description,
        "criterio_seleccion": (
            f"Mayor {SELECTION_METRIC} medio en validación cruzada estratificada de "
            f"{CV_FOLDS} pliegues sobre el conjunto de entrenamiento. El conjunto de "
            "prueba solo se usa para la evaluación final informativa."
        ),
        "modelo_seleccionado": winner,
        "ranking_cv": ranking,
        "modelos": res.results,
        "importancia_permutacion_prueba": importances,
        "generado_utc": generated,
    }
    paths["metricas"].write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    X_train, y_train = res.X_train, res.y_train
    metadata = {
        "proyecto": "EduPredict AI",
        "modalidad": fs.name,
        "descripcion": fs.description,
        "archivo_modelo": paths["modelo"].name,
        "algoritmo": winner,
        "algoritmo_nombre": MODEL_NAMES[winner],
        "hiperparametros": res.results[winner]["mejores_hiperparametros"],
        "clases": [str(c) for c in best_pipe.classes_],
        "columnas_entrada": fs.raw_columns,
        "columnas_nominales": list(fs.nominal),
        "columnas_binarias": list(fs.binary),
        "columnas_numericas": list(fs.numeric),
        "columnas_derivadas": fs.engineered,
        "categorias_entrenamiento": {c: sorted(int(v) for v in X_train[c].unique()) for c in fs.nominal},
        "modas_entrenamiento": {c: int(X_train[c].mode().iloc[0]) for c in list(fs.nominal) + list(fs.binary)},
        "estadisticas_numericas_entrenamiento": {
            c: {"min": _param_repr(X_train[c].min()), "max": _param_repr(X_train[c].max()),
                "mediana": _param_repr(X_train[c].median())}
            for c in fs.numeric
        },
        "particion": {
            "test_size": TEST_SIZE,
            "estratificada": True,
            "random_state": RANDOM_STATE,
            "n_entrenamiento": int(len(X_train)),
            "n_prueba": int(len(res.X_test)),
            "distribucion_entrenamiento": {k: int(v) for k, v in y_train.value_counts().items()},
            "distribucion_prueba": {k: int(v) for k, v in res.y_test.value_counts().items()},
        },
        "validacion_cruzada": {"tipo": "StratifiedKFold", "n_splits": CV_FOLDS, "shuffle": True,
                               "random_state": RANDOM_STATE, "metrica_refit": SELECTION_METRIC},
        "dataset": {
            "nombre": "Predict Students' Dropout and Academic Success",
            "uci_id": 697,
            "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
            "doi": "10.24432/C5MC89",
            "archivo": "data/data.csv",
            "sha256": file_sha256(RAW_PATH) if RAW_PATH.exists() else None,
            "filas": int(len(df)),
        },
        "entorno": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "regenerar": "python -m src.train",
        "generado_utc": generated,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[%s] Modelo seleccionado: %s -> %s", fs.name, winner, paths["modelo"])
    return metricas


def train_modality(df: pd.DataFrame, fs: FeatureSet, out_dir: Path = MODELS_DIR, n_jobs: int = -1) -> dict:
    """Atajo: ``fit_modality`` + ``export_modality``."""
    return export_modality(fit_modality(df, fs, n_jobs=n_jobs), df, out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Entrena los modelos de EduPredict AI")
    parser.add_argument("--download", action="store_true", help="Descarga el CSV oficial de UCI si no existe")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Procesos para GridSearchCV")
    parser.add_argument("--out", type=Path, default=MODELS_DIR, help="Carpeta de salida de artefactos")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_raw(download_if_missing=args.download)
    df = clean_dataset(raw, strict=True)
    save_clean(df)
    logger.info("Datos limpios: %s | consistencia: %s", df.shape, consistency_report(df))

    summary = {}
    for fs in (FULL, EARLY):
        m = train_modality(df, fs, args.out, n_jobs=args.n_jobs)
        w = m["modelo_seleccionado"]
        summary[fs.name] = {
            "modelo": w,
            "hiperparametros": m["modelos"][w]["mejores_hiperparametros"],
            "cv_f1_macro": m["modelos"][w]["cv"]["f1_macro"]["media"],
            "prueba_f1_macro": m["modelos"][w]["prueba"]["f1_macro"],
            "prueba_accuracy": m["modelos"][w]["prueba"]["accuracy"],
            "prueba_balanced_accuracy": m["modelos"][w]["prueba"]["balanced_accuracy"],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
