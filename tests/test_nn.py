"""Pruebas del codificador y de las arquitecturas neuronales del ACA 2."""

import numpy as np
import pytest

from src import data_dictionary as dd
from src.data_loader import load_clean
from src.nn.data import TabularEncoder
from src.predict import ModelNotAvailableError, check_artifacts, load_model
from src.preprocessing import EARLY, FULL
from src.train import split_data


@pytest.fixture(scope="module")
def df():
    return load_clean()


@pytest.mark.parametrize("fs", [FULL, EARLY])
def test_encoder_shapes_and_no_nans(df, fs):
    Xtr, Xte, _, _ = split_data(df, fs)
    enc = TabularEncoder(fs).fit(Xtr)
    xn, xc = enc.transform(Xte)
    assert xn.shape == (len(Xte), len(fs.binary) + len(fs.numeric_with_engineered))
    assert xc.shape == (len(Xte), len(fs.nominal))
    assert np.isfinite(xn).all() and xn.dtype == np.float32 and xc.dtype == np.int64
    assert xn.max() <= 5 and xn.min() >= -5
    assert (xc.max(axis=0) < np.array(enc.cardinalities)).all() and xc.min() >= 0


def test_encoder_learns_only_from_train(df):
    """Las medias se calculan solo con el conjunto de ajuste (sin fuga de prueba)."""
    Xtr, Xte, _, _ = split_data(df, FULL)
    enc = TabularEncoder(FULL).fit(Xtr)
    full_mean = TabularEncoder(FULL).fit(df[FULL.raw_columns]).mean_
    assert not np.allclose(enc.mean_, full_mean)


def test_unseen_category_maps_to_zero(df):
    Xtr, _, _, _ = split_data(df, FULL)
    enc = TabularEncoder(FULL).fit(Xtr)
    row = Xtr.iloc[[0]].copy()
    row["course"] = 424242
    _, xc = enc.transform(row)
    assert xc[0, enc.cat_cols.index("course")] == 0


def test_early_encoder_excludes_second_semester(df):
    enc = TabularEncoder(EARLY).fit(split_data(df, EARLY)[0])
    assert not any("2nd" in c for c in enc.num_cols)


def test_encoder_roundtrip(df):
    Xtr, Xte, _, _ = split_data(df, FULL)
    enc = TabularEncoder(FULL).fit(Xtr)
    enc2 = TabularEncoder.from_dict(enc.to_dict())
    a, b = enc.transform(Xte), enc2.transform(Xte)
    assert np.allclose(a[0], b[0]) and (a[1] == b[1]).all()


def test_missing_artifacts_raise(tmp_path):
    with pytest.raises(ModelNotAvailableError):
        load_model("completo", tmp_path)
    assert check_artifacts(tmp_path)["completo"]


# ---- requieren PyTorch ------------------------------------------------------


@pytest.mark.parametrize("arch,params", [
    ("mlp", {"emb_dim": 4, "hidden": [16], "dropout": 0.1}),
    ("ft_transformer", {"d_token": 16, "n_layers": 1, "n_heads": 2, "ffn_mult": 2, "dropout": 0.1}),
])
def test_forward_shapes(df, arch, params):
    torch = pytest.importorskip("torch")
    from src.nn.models import build_model
    Xtr, _, _, _ = split_data(df, FULL)
    enc = TabularEncoder(FULL).fit(Xtr)
    xn, xc = enc.transform(Xtr.iloc[:8])
    model = build_model(arch, enc.n_num, enc.cardinalities, 3, params)
    out = model(torch.from_numpy(xn), torch.from_numpy(xc))
    assert out.shape == (8, 3)


def test_short_training_save_load_roundtrip(df, tmp_path):
    pytest.importorskip("torch")
    """Entrena 3 épocas, guarda, recarga y verifica predicciones idénticas."""
    from src.nn.training import fit_with_inner_val
    from src.nn.wrapper import NNClassifier
    Xtr, Xte, ytr, _ = split_data(df, EARLY)
    cfg = {"arch": {"emb_dim": 4, "hidden": [16], "dropout": 0.1}, "train": {"lr": 1e-3}}
    clf, hist = fit_with_inner_val("mlp", cfg, EARLY, Xtr.iloc[:600], ytr.iloc[:600], max_epochs=3, patience=3)
    assert 1 <= hist["best_epoch"] <= 3 and len(hist["val_loss"]) == hist["epochs_run"]
    p = clf.predict_proba(Xte.iloc[:20])
    assert p.shape == (20, 3) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    clf.save(tmp_path / "m.pt")
    clf2 = NNClassifier.load(tmp_path / "m.pt")
    assert np.allclose(p, clf2.predict_proba(Xte.iloc[:20]), atol=1e-6)
    assert list(clf2.classes_) == dd.CLASSES
