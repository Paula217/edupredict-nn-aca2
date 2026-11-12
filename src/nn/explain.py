"""Importancia por permutación para cualquier clasificador con ``predict``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import data_dictionary as dd


def permutation_importances(clf, X: pd.DataFrame, y, n_repeats: int = 10,
                            random_state: int = 42, top: int = 15) -> list[dict]:
    """Caída media de macro-F1 al barajar cada variable original (misma salida que ``src.evaluate``)."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    base = f1_score(y, clf.predict(X), average="macro")
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base - f1_score(y, clf.predict(Xp), average="macro"))
        rows.append({"variable": col, "etiqueta": dd.label(col),
                     "importancia_media": float(np.mean(drops)), "importancia_desv": float(np.std(drops))})
    rows.sort(key=lambda r: r["importancia_media"], reverse=True)
    return rows[:top]
