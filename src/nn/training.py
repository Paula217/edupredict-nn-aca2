"""Entrenamiento, validación cruzada y exportación de las redes neuronales.

Protocolo (idéntico al del ACA 1 para que los resultados sean comparables):

* Partición estratificada 80/20, ``RANDOM_STATE = 42`` (``src.train.split_data``).
* Validación cruzada estratificada de 5 pliegues sobre el 80 % de entrenamiento;
  métrica de selección: **macro-F1**.
* En cada pliegue el codificador se ajusta solo con los datos de entrenamiento
  del pliegue y, dentro de ellos, se aparta un 15 % para *early stopping*.
* Desbalance: pérdida de entropía cruzada con **pesos de clase** balanceados.
"""

from __future__ import annotations

import copy
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

from src import data_dictionary as dd
from src.evaluate import compute_metrics
from src.nn.data import TabularEncoder
from src.nn.explain import permutation_importances
from src.nn.models import ARCH_NAMES, build_model, count_parameters
from src.nn.wrapper import NNClassifier, proba_from_arrays
from src.preprocessing import FeatureSet
from src.train import CV_FOLDS, RANDOM_STATE, TEST_SIZE, file_sha256

CLASSES = list(dd.CLASSES)
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
INNER_VAL_SIZE = 0.15
NN_ARTIFACTS = {
    "completo": ("modelo_nn.pt", "metadata_nn.json", "metricas_nn.json"),
    "temprano": ("modelo_nn_temprano.pt", "metadata_nn_temprano.json", "metricas_nn_temprano.json"),
}

# Configuraciones candidatas (se comparan por macro-F1 en validación cruzada)
DEFAULT_TRAIN = {"lr": 1e-3, "weight_decay": 1e-2, "batch_size": 128}
GRIDS = {
    "mlp": [
        {"arch": {"emb_dim": 4, "hidden": [64], "dropout": 0.3}, "train": {"lr": 1e-3, "weight_decay": 1e-2}},
        {"arch": {"emb_dim": 4, "hidden": [128, 64], "dropout": 0.4}, "train": {"lr": 1e-3, "weight_decay": 1e-2}},
        {"arch": {"emb_dim": 8, "hidden": [64], "dropout": 0.5}, "train": {"lr": 1e-3, "weight_decay": 5e-2}},
        {"arch": {"emb_dim": 4, "hidden": [32], "dropout": 0.3}, "train": {"lr": 2e-3, "weight_decay": 1e-2}},
    ],
    "ft_transformer": [
        {"arch": {"d_token": 32, "n_layers": 2, "n_heads": 4, "ffn_mult": 2, "dropout": 0.2},
         "train": {"lr": 5e-4, "weight_decay": 1e-2}},
        {"arch": {"d_token": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.3},
         "train": {"lr": 1e-3, "weight_decay": 1e-2}},
    ],
}


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_labels(y) -> np.ndarray:
    return np.asarray([CLASS_TO_INT[str(v)] for v in y], dtype=np.int64)


def class_weights(y_int: np.ndarray) -> torch.Tensor:
    """Pesos balanceados: n / (k · n_c) — equivalente a ``class_weight='balanced'`` de scikit-learn."""
    counts = np.bincount(y_int, minlength=len(CLASSES)).astype(float)
    return torch.tensor(len(y_int) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)


def fit_nn(arch: str, arch_params: dict, train_params: dict, enc: TabularEncoder,
           Xtr: pd.DataFrame, ytr, Xva: pd.DataFrame, yva,
           seed: int = RANDOM_STATE, max_epochs: int = 150, patience: int = 20):
    """Entrena una red con *early stopping* sobre el macro-F1 de validación.

    Devuelve ``(modelo_con_mejores_pesos, historial)``. El historial contiene,
    por época, pérdida / accuracy / macro-F1 de entrenamiento y validación
    (ambas medidas en modo evaluación, sin dropout, para curvas comparables).
    """
    set_seed(seed)
    tp = {**DEFAULT_TRAIN, **train_params}
    xn_tr, xc_tr = enc.transform(Xtr)
    xn_va, xc_va = enc.transform(Xva)
    y_tr, y_va = encode_labels(ytr), encode_labels(yva)

    model = build_model(arch, enc.n_num, enc.cardinalities, len(CLASSES), arch_params)
    crit = nn.NLLLoss(weight=class_weights(y_tr))          # entrada: log-probabilidades
    ce = nn.CrossEntropyLoss(weight=class_weights(y_tr))   # para el paso de entrenamiento (logits)
    opt = torch.optim.AdamW(model.parameters(), lr=tp["lr"], weight_decay=tp["weight_decay"])

    t_xn, t_xc, t_y = torch.from_numpy(xn_tr), torch.from_numpy(xc_tr), torch.from_numpy(y_tr)
    gen = torch.Generator().manual_seed(seed)
    n, bs = len(y_tr), int(tp["batch_size"])

    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1")}
    best_f1, best_state, best_epoch, wait = -1.0, None, 0, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = ce(model(t_xn[idx], t_xc[idx]), t_y[idx])
            loss.backward()
            opt.step()
        tl, ta, tf = _eval_with(model, crit, xn_tr, xc_tr, y_tr)
        vl, va, vf = _eval_with(model, crit, xn_va, xc_va, y_va)
        for k, v in zip(hist, (tl, vl, ta, va, tf, vf)):
            hist[k].append(v)
        if vf > best_f1 + 1e-4:
            best_f1, best_epoch, wait = vf, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist["best_epoch"] = best_epoch
    hist["epochs_run"] = len(hist["val_f1"])
    return model, hist


def _eval_with(model, crit, xn, xc, y_int):
    """Pérdida ponderada (NLL sobre log-softmax), accuracy y macro-F1."""
    proba = proba_from_arrays(model, xn, xc)
    logp = torch.from_numpy(np.log(np.clip(proba, 1e-12, 1.0)).astype(np.float32))
    loss = float(crit(logp, torch.from_numpy(y_int)))
    pred = proba.argmax(1)
    return loss, float(accuracy_score(y_int, pred)), float(f1_score(y_int, pred, average="macro"))


def fit_with_inner_val(arch, cfg, fs: FeatureSet, X: pd.DataFrame, y, seed=RANDOM_STATE, **kw):
    """Ajusta el codificador con ``X`` y entrena apartando 15 % para early stopping."""
    enc = TabularEncoder(fs).fit(X)
    X1, X2, y1, y2 = train_test_split(X, y, test_size=INNER_VAL_SIZE, stratify=y, random_state=seed)
    model, hist = fit_nn(arch, cfg["arch"], cfg.get("train", {}), enc, X1, y1, X2, y2, seed=seed, **kw)
    clf = NNClassifier(model, enc, CLASSES, arch, cfg["arch"], {**DEFAULT_TRAIN, **cfg.get("train", {})})
    return clf, hist


def cross_validate(arch: str, cfg: dict, fs: FeatureSet, X: pd.DataFrame, y, seed=RANDOM_STATE, **kw) -> dict:
    """Validación cruzada estratificada de 5 pliegues (mismo esquema que el ACA 1)."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y = pd.Series(np.asarray(y), index=X.index)
    names = ["accuracy", "balanced_accuracy", "f1_macro"] + [f"recall_{c}" for c in CLASSES]
    folds = {k: [] for k in names}
    for tr, va in skf.split(X, y):
        clf, _ = fit_with_inner_val(arch, cfg, fs, X.iloc[tr], y.iloc[tr], seed=seed, **kw)
        pred = clf.predict(X.iloc[va])
        yv = y.iloc[va]
        folds["accuracy"].append(float(accuracy_score(yv, pred)))
        folds["balanced_accuracy"].append(float(balanced_accuracy_score(yv, pred)))
        folds["f1_macro"].append(float(f1_score(yv, pred, average="macro")))
        for c in CLASSES:
            folds[f"recall_{c}"].append(float(recall_score(yv, pred, labels=[c], average="macro", zero_division=0)))
    return {k: {"media": float(np.mean(v)), "desv": float(np.std(v)), "pliegues": v} for k, v in folds.items()}


def select_config(arch: str, fs: FeatureSet, X: pd.DataFrame, y, grid=None, verbose=True, **kw):
    """Evalúa cada configuración por CV y devuelve (mejor_cfg, cv_mejor, tabla)."""
    grid = grid or GRIDS[arch]
    rows, best = [], None
    for i, cfg in enumerate(grid):
        cv = cross_validate(arch, cfg, fs, X, y, **kw)
        f = cv["f1_macro"]
        rows.append({"config": i, "arquitectura": json.dumps(cfg["arch"]), "entrenamiento": json.dumps(cfg.get("train", {})),
                     "cv_f1_macro": f["media"], "cv_desv": f["desv"]})
        if verbose:
            print(f"[{fs.name}] {arch} cfg{i}: macro-F1 CV = {f['media']:.4f} ± {f['desv']:.4f}")
        if best is None or f["media"] > best[1]["f1_macro"]["media"]:
            best = (cfg, cv)
    return best[0], best[1], pd.DataFrame(rows)


def evaluate_on_test(clf: NNClassifier, X_test: pd.DataFrame, y_test) -> dict:
    return compute_metrics(y_test, clf.predict(X_test), clf.predict_proba(X_test), classes=CLASSES)


def history_to_json(hist: dict) -> dict:
    return {k: ([float(x) for x in v] if isinstance(v, list) else int(v)) for k, v in hist.items()}


def export_modality(fs: FeatureSet, winner: str, clf: NNClassifier, results: dict, baselines: dict,
                    X_train, X_test, y_train, y_test, df: pd.DataFrame, out_dir: Path) -> dict:
    """Guarda ``modelo_nn*.pt``, ``metadata_nn*.json`` y ``metricas_nn*.json`` (formato compatible con la app).

    ``results``  : {arch: {"nombre", "mejores_hiperparametros", "cv", "prueba", "historial", "parametros"}}
    ``baselines``: modelos clásicos del ACA 1 (logreg, rf, hgb) con su CV y prueba, para la comparación.
    """
    f_model, f_meta, f_metrics = NN_ARTIFACTS[fs.name]
    out_dir = Path(out_dir)
    clf.save(out_dir / f_model)
    all_models = {**results, **baselines}
    ranking = sorted(all_models, key=lambda k: all_models[k]["cv"]["f1_macro"]["media"], reverse=True)
    importances = permutation_importances(clf, X_test, y_test)
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")

    metricas = {
        "modalidad": fs.name,
        "descripcion": fs.description,
        "criterio_seleccion": (
            f"Entre las redes neuronales (MLP y FT-Transformer) se despliega la de mayor macro-F1 medio en validación "
            f"cruzada estratificada de {CV_FOLDS} pliegues sobre el conjunto de entrenamiento. Los modelos clásicos del "
            "ACA 1 se incluyen como referencia. El conjunto de prueba solo se usa para la evaluación final."
        ),
        "modelo_seleccionado": winner,
        "ranking_cv": ranking,
        "modelos": all_models,
        "importancia_permutacion_prueba": importances,
        "generado_utc": generated,
    }
    (out_dir / f_metrics).write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "proyecto": "EduPredict AI",
        "modalidad": fs.name,
        "descripcion": fs.description,
        "archivo_modelo": f_model,
        "algoritmo": winner,
        "algoritmo_nombre": ARCH_NAMES[winner],
        "hiperparametros": {"arquitectura": clf.arch_params, "entrenamiento": clf.train_params},
        "clases": CLASSES,
        "columnas_entrada": fs.raw_columns,
        "columnas_nominales": list(fs.nominal),
        "columnas_binarias": list(fs.binary),
        "columnas_numericas": list(fs.numeric),
        "columnas_derivadas": fs.engineered,
        "categorias_entrenamiento": {c: sorted(int(v) for v in X_train[c].unique()) for c in fs.nominal},
        "modas_entrenamiento": {c: int(X_train[c].mode().iloc[0]) for c in list(fs.nominal) + list(fs.binary)},
        "estadisticas_numericas_entrenamiento": {
            c: {"min": float(X_train[c].min()), "max": float(X_train[c].max()), "mediana": float(X_train[c].median())}
            for c in fs.numeric
        },
        "particion": {
            "test_size": TEST_SIZE, "estratificada": True, "random_state": RANDOM_STATE,
            "n_entrenamiento": int(len(X_train)), "n_prueba": int(len(X_test)),
            "distribucion_entrenamiento": {k: int(v) for k, v in pd.Series(y_train).value_counts().items()},
            "distribucion_prueba": {k: int(v) for k, v in pd.Series(y_test).value_counts().items()},
        },
        "validacion_cruzada": {"tipo": "StratifiedKFold", "n_splits": CV_FOLDS, "shuffle": True,
                               "random_state": RANDOM_STATE, "metrica_seleccion": "f1_macro"},
        "parametros_entrenables": results[winner].get("parametros"),
        "dataset": {
            "nombre": "Predict Students' Dropout and Academic Success", "uci_id": 697,
            "url": "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success",
            "doi": "10.24432/C5MC89", "archivo": "data/data.csv",
            "sha256": file_sha256(Path(__file__).resolve().parents[2] / "data" / "data.csv"),
            "filas": int(len(df)),
        },
        "entorno": {"python": platform.python_version(), "torch": torch.__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
        "regenerar": "Ejecutar notebooks/ACA2_Student_Dropout_NN.ipynb",
        "generado_utc": generated,
    }
    (out_dir / f_meta).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metricas


def n_parameters(clf: NNClassifier) -> int:
    return count_parameters(clf.model)
