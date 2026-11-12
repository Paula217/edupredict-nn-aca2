"""Métricas de evaluación multiclase y utilidades de interpretación."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from src import data_dictionary as dd


def cv_scoring() -> dict:
    """Métricas calculadas en cada pliegue de la validación cruzada."""
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    for cls in dd.CLASSES:
        scoring[f"recall_{cls}"] = make_scorer(recall_score, labels=[cls], average="macro", zero_division=0)
    return scoring


def compute_metrics(y_true, y_pred, y_proba: np.ndarray | None = None,
                    classes: list[str] | None = None) -> dict:
    """Calcula las métricas exigidas por el ACA sobre un conjunto de evaluación.

    Incluye accuracy, balanced accuracy, macro-F1, precisión/recall/F1 por
    clase, soporte, matriz de confusión (filas = real, columnas = predicho) y,
    si hay probabilidades, ROC-AUC macro uno-contra-resto.
    """
    classes = list(classes or dd.CLASSES)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)),
        "por_clase": {
            cls: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "soporte": int(s[i]),
            }
            for i, cls in enumerate(classes)
        },
        "matriz_confusion": {
            "etiquetas": classes,
            "valores": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        },
    }
    if y_proba is not None:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(pd.Series(y_true), y_proba, multi_class="ovr", average="macro", labels=classes)
        )
    return out


def summarize_cv(search, scoring_names: list[str]) -> dict:
    """Media y desviación de CV para la mejor combinación.

    Incluye la media en los pliegues de entrenamiento (para medir la brecha de
    sobreajuste train–validación, como en el notebook borrador) y los valores
    de cada pliegue de validación (estabilidad).
    """
    i = int(search.best_index_)
    res = search.cv_results_
    n_splits = sum(1 for k in res if k.startswith("split") and k.endswith(f"_test_{scoring_names[0]}"))
    out = {}
    for name in scoring_names:
        out[name] = {
            "media": float(res[f"mean_test_{name}"][i]),
            "desv": float(res[f"std_test_{name}"][i]),
            "pliegues": [float(res[f"split{k}_test_{name}"][i]) for k in range(n_splits)],
        }
        if f"mean_train_{name}" in res:
            out[name]["media_train"] = float(res[f"mean_train_{name}"][i])
    return out


def permutation_importances(pipeline, X: pd.DataFrame, y, n_repeats: int = 10,
                            random_state: int = 42, top: int = 15) -> list[dict]:
    """Importancia por permutación (caída de macro-F1) sobre columnas originales.

    Se calcula sobre el pipeline completo, de modo que cada variable original
    (p. ej. ``course``) se permuta como un todo, sin importar cuántas columnas
    One-Hot genere. Es una medida de asociación del modelo, no de causalidad.
    """
    result = permutation_importance(
        pipeline, X, y, scoring="f1_macro", n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1,
    )
    rows = [
        {
            "variable": col,
            "etiqueta": dd.label(col),
            "importancia_media": float(result.importances_mean[i]),
            "importancia_desv": float(result.importances_std[i]),
        }
        for i, col in enumerate(X.columns)
    ]
    rows.sort(key=lambda r: r["importancia_media"], reverse=True)
    return rows[:top]
