"""``NNClassifier``: red entrenada + codificador, con interfaz tipo scikit-learn.

La aplicación Streamlit y las pruebas usan ``predict_proba`` / ``predict`` /
``classes_`` igual que con los pipelines de scikit-learn del ACA 1. El archivo
``.pt`` contiene únicamente tensores y tipos primitivos (se carga con
``torch.load(..., weights_only=True)``: no ejecuta código arbitrario).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.nn.data import TabularEncoder
from src.nn.models import build_model


def proba_from_arrays(model: torch.nn.Module, x_num: np.ndarray, x_cat: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Probabilidades softmax (modo evaluación, sin dropout, sin gradientes)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_num), batch):
            xn = torch.from_numpy(x_num[i:i + batch])
            xc = torch.from_numpy(x_cat[i:i + batch])
            out.append(torch.softmax(model(xn, xc), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


class NNClassifier:
    def __init__(self, model: torch.nn.Module, enc: TabularEncoder, classes: list[str],
                 arch: str, arch_params: dict, train_params: dict | None = None):
        self.model = model
        self.enc = enc
        self.classes_ = np.asarray(classes)
        self.arch = arch
        self.arch_params = dict(arch_params)
        self.train_params = dict(train_params or {})

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        x_num, x_cat = self.enc.transform(X)
        return proba_from_arrays(self.model, x_num, x_cat)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # -- persistencia ------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "arch": self.arch,
            "arch_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.arch_params.items()},
            "train_params": self.train_params,
            "classes": [str(c) for c in self.classes_],
            "encoder": self.enc.to_dict(),
            "state_dict": self.model.state_dict(),
        }, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "NNClassifier":
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        enc = TabularEncoder.from_dict(blob["encoder"])
        model = build_model(blob["arch"], enc.n_num, enc.cardinalities, len(blob["classes"]), blob["arch_params"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return cls(model, enc, blob["classes"], blob["arch"], blob["arch_params"], blob.get("train_params"))
