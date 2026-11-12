"""Pruebas de artefactos, predicción y arranque de la aplicación."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import data_dictionary as dd
from src.data_loader import load_clean
from src.predict import (
    ModelNotAvailableError, artifact_paths, check_artifacts, describe_prediction,
    load_metadata, load_metrics, load_model, predict_frame, predict_one,
)
from src.preprocessing import EARLY, FULL, SchemaError

# Estas pruebas necesitan los modelos entrenados (models/*.pt), que se generan al ejecutar el notebook del ACA 2.
pytestmark = pytest.mark.skipif(
    any(check_artifacts().values()), reason="Faltan models/*.pt: ejecute notebooks/ACA2_Student_Dropout_NN.ipynb"
)
NN_KEYS = {"mlp", "ft_transformer"}

MODALITIES = ["completo", "temprano"]
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(scope="module")
def data():
    return load_clean()


def test_artifacts_exist():
    assert check_artifacts() == {"completo": [], "temprano": []}


@pytest.mark.parametrize("modality", MODALITIES)
def test_metadata_and_metrics_are_consistent(modality):
    meta, met = load_metadata(modality), load_metrics(modality)
    fs = FULL if modality == "completo" else EARLY
    assert meta["columnas_entrada"] == fs.raw_columns
    assert meta["clases"] == dd.CLASSES
    assert met["modelo_seleccionado"] == meta["algoritmo"]
    assert set(met["modelos"]) == NN_KEYS | {"logreg", "rf", "hgb"}
    # El ganador es la red de mayor macro-F1 en CV (no en prueba)
    cv = {k: met["modelos"][k]["cv"]["f1_macro"]["media"] for k in NN_KEYS}
    assert met["modelo_seleccionado"] == max(cv, key=cv.get)
    assert met["modelos"][met["modelo_seleccionado"]]["historial"]["best_epoch"] >= 1
    for v in met["modelos"].values():
        cm = np.array(v["prueba"]["matriz_confusion"]["valores"])
        assert cm.sum() == meta["particion"]["n_prueba"]
        assert 0 <= v["prueba"]["f1_macro"] <= 1


def test_early_model_does_not_use_second_semester():
    meta = load_metadata("temprano")
    assert not any(c.startswith("curricular_units_2nd") for c in meta["columnas_entrada"] + meta["columnas_derivadas"])
    model = load_model("temprano")
    assert not any("2nd" in c for c in model.enc.num_cols)


@pytest.mark.parametrize("modality", MODALITIES)
def test_predict_one_returns_valid_probabilities(data, modality):
    fs = FULL if modality == "completo" else EARLY
    rec = data[fs.raw_columns].iloc[0].to_dict()
    res = predict_one(rec, modality)
    assert res.clase in dd.CLASSES
    assert abs(sum(res.probabilidades.values()) - 1) < 1e-6
    assert res.clase == max(res.probabilidades, key=res.probabilidades.get)
    assert isinstance(describe_prediction(res), str)


def test_predict_frame_matches_pipeline(data):
    """La ruta de la interfaz produce exactamente lo mismo que el pipeline directo."""
    X = data[FULL.raw_columns].sample(50, random_state=0)
    out, _ = predict_frame(X, "completo")
    direct = load_model("completo").predict_proba(X)
    assert np.allclose(out[[f"p_{c}" for c in dd.CLASSES]].values, direct)


def test_form_like_types_are_accepted(data):
    """El formulario envía floats/ints de Python; deben dar el mismo resultado que el CSV."""
    rec = data[FULL.raw_columns].iloc[5].to_dict()
    as_float = {k: float(v) for k, v in rec.items()}
    a = predict_one(rec, "completo").probabilidades
    b = predict_one(as_float, "completo").probabilidades
    assert a == pytest.approx(b)


def test_unseen_category_predicts_with_warning(data):
    rec = data[FULL.raw_columns].iloc[0].to_dict()
    rec["course"] = 4242
    res = predict_one(rec, "completo")
    assert res.advertencias
    assert abs(sum(res.probabilidades.values()) - 1) < 1e-6


def test_invalid_input_raises(data):
    rec = data[FULL.raw_columns].iloc[0].to_dict()
    rec["curricular_units_1st_sem_grade"] = 30
    with pytest.raises(SchemaError):
        predict_one(rec, "completo")
    rec = data[EARLY.raw_columns].iloc[0].to_dict()
    with pytest.raises(SchemaError):
        predict_one(rec, "completo")  # faltan columnas del 2.º semestre


def test_metrics_reproducible_from_artifact(data):
    """Recalcular accuracy en la partición de prueba reproduce lo guardado en metricas.json."""
    from sklearn.model_selection import train_test_split
    meta, met = load_metadata("completo"), load_metrics("completo")
    X, y = data[FULL.raw_columns], data[dd.TARGET]
    _, X_te, _, y_te = train_test_split(X, y, test_size=meta["particion"]["test_size"], stratify=y,
                                        random_state=meta["particion"]["random_state"])
    acc = (load_model("completo").predict(X_te) == y_te).mean()
    assert acc == pytest.approx(met["modelos"][met["modelo_seleccionado"]]["prueba"]["accuracy"])


def test_json_files_are_valid_utf8():
    for mod in MODALITIES:
        for key in ("metadata", "metricas"):
            json.loads(artifact_paths(mod)[key].read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Arranque de Streamlit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("page", ["Dashboard", "Predicción individual", "Comparación de modelos", "Acerca del proyecto"])
def test_streamlit_pages_run_without_exceptions(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception


def test_streamlit_prediction_flow():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Predicción individual").run()
    for modality_label in at.radio[0].options:
        at.radio[0].set_value(modality_label).run()
        submit = [b for b in at.button if b.label == "Analizar estudiante"][0]
        submit.click().run()
        assert not at.exception, at.exception
        assert not at.error
        assert any("Herramienta académica" in i.value for i in at.info)
