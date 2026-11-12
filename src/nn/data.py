"""Codificación tabular para redes neuronales (sin PyTorch).

Convierte el DataFrame crudo en dos matrices:

* ``x_num`` (float32): variables numéricas y binarias estandarizadas +
  las tasas de aprobación por semestre (ingeniería de variables sin estado).
* ``x_cat`` (int64): índice de categoría de cada variable nominal. El índice
  **0** está reservado para categorías no vistas o infrecuentes
  (< ``MIN_CATEGORY_FREQUENCY`` apariciones en entrenamiento); alimenta una
  capa de *embeddings*.

Todo lo que se aprende (medias, desviaciones, categorías) se calcula en
``fit`` solo con datos de entrenamiento -> sin fuga de información. El
codificador se serializa como diccionario simple (JSON-compatible).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import FEATURE_SETS, MIN_CATEGORY_FREQUENCY, AcademicFeatureEngineer, FeatureSet

CLIP = 5.0  # recorte de z-scores para amortiguar valores extremos (sin eliminar filas)


class TabularEncoder:
    def __init__(self, fs: FeatureSet, min_frequency: int = MIN_CATEGORY_FREQUENCY):
        self.fs = fs
        self.min_frequency = min_frequency
        self.engineer = AcademicFeatureEngineer(semesters=fs.semesters)
        self.cat_cols = list(fs.nominal)
        self.num_cols = list(fs.binary) + list(fs.numeric_with_engineered)
        self.categories_: dict[str, list[int]] = {}
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # -- ajuste -----------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "TabularEncoder":
        Xf = self.engineer.transform(X[self.fs.raw_columns])
        for c in self.cat_cols:
            counts = Xf[c].astype(int).value_counts()
            self.categories_[c] = sorted(int(k) for k, v in counts.items() if v >= self.min_frequency)
        vals = Xf[self.num_cols].astype(float).to_numpy()
        self.mean_ = vals.mean(axis=0)
        std = vals.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    # -- transformación ----------------------------------------------------
    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xf = self.engineer.transform(X[self.fs.raw_columns])
        x_num = (Xf[self.num_cols].astype(float).to_numpy() - self.mean_) / self.std_
        x_num = np.clip(x_num, -CLIP, CLIP).astype(np.float32)
        cols = []
        for c in self.cat_cols:
            lookup = {code: i + 1 for i, code in enumerate(self.categories_[c])}
            cols.append(Xf[c].astype(int).map(lambda v, lk=lookup: lk.get(int(v), 0)).to_numpy())
        x_cat = np.stack(cols, axis=1).astype(np.int64)
        return x_num, x_cat

    def fit_transform(self, X: pd.DataFrame):
        return self.fit(X).transform(X)

    # -- metadatos ---------------------------------------------------------
    @property
    def n_num(self) -> int:
        return len(self.num_cols)

    @property
    def cardinalities(self) -> list[int]:
        """Tamaño del vocabulario de cada variable nominal (incluye el índice 0 = desconocido)."""
        return [len(self.categories_[c]) + 1 for c in self.cat_cols]

    # -- serialización -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "modalidad": self.fs.name,
            "min_frequency": int(self.min_frequency),
            "categories": {c: [int(v) for v in vs] for c, vs in self.categories_.items()},
            "mean": [float(v) for v in self.mean_],
            "std": [float(v) for v in self.std_],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TabularEncoder":
        enc = cls(FEATURE_SETS[d["modalidad"]], d.get("min_frequency", MIN_CATEGORY_FREQUENCY))
        enc.categories_ = {c: [int(v) for v in vs] for c, vs in d["categories"].items()}
        enc.mean_ = np.asarray(d["mean"], dtype=float)
        enc.std_ = np.asarray(d["std"], dtype=float)
        return enc
