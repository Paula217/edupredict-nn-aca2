"""Pruebas de carga, limpieza y preprocesamiento."""

import numpy as np
import pandas as pd
import pytest

from src import data_dictionary as dd
from src.data_loader import (
    RAW_PATH, DataValidationError, clean_column_names, clean_dataset, consistency_report, load_clean, load_raw,
    normalizar_nombre, soft_rule_flags,
)
from src.preprocessing import (
    EARLY, FULL, AcademicFeatureEngineer, SchemaError, WinsorizadorIQR, build_preprocessor, validate_input,
)


@pytest.fixture(scope="module")
def raw():
    return load_raw(RAW_PATH)


@pytest.fixture(scope="module")
def clean(raw):
    return clean_dataset(raw)


def test_raw_shape_matches_uci(raw):
    assert raw.shape == (4424, 37)


def test_column_names_are_cleaned(raw):
    # El CSV original trae BOM y un tabulador en "Daytime/evening attendance\t"
    out = clean_column_names(raw)
    assert list(out.columns) == list(dd.COLUMN_MAP.values())
    assert not any("\t" in c or "﻿" in c for c in out.columns)


def test_clean_dataset_integrity(clean):
    assert clean.shape == (4424, 37)
    assert clean.isna().sum().sum() == 0
    assert set(clean[dd.TARGET]) == set(dd.CLASSES)
    rep = consistency_report(clean)
    assert rep["fuera_de_rango"] == {}
    assert rep["reglas"]["aprobadas_mayor_inscritas_1st"] == 0


def test_no_outliers_removed(clean):
    # Las edades extremas son legítimas y deben conservarse
    assert clean["age_at_enrollment"].max() == 70


def test_clean_csv_is_consistent_with_pipeline(clean):
    """El CSV versionado debe coincidir exactamente con la limpieza del pipeline."""
    stored = load_clean()
    pd.testing.assert_frame_equal(stored.reset_index(drop=True), clean.reset_index(drop=True), check_dtype=False)


def test_strict_validation_rejects_bad_data(raw):
    bad = raw.copy()
    bad.loc[0, "Target"] = "Unknown"
    with pytest.raises(DataValidationError):
        clean_dataset(bad)
    missing = raw.drop(columns=["Course"])
    with pytest.raises(DataValidationError):
        clean_dataset(missing)


def test_feature_sets():
    assert len(FULL.raw_columns) == 36
    assert not any(c.startswith("curricular_units_2nd") for c in EARLY.raw_columns)
    assert len(EARLY.raw_columns) == 30
    assert EARLY.engineered == ["curricular_units_1st_sem_approval_rate"]


def test_feature_engineer_handles_zero_enrolled():
    X = pd.DataFrame({"curricular_units_1st_sem_enrolled": [0, 4], "curricular_units_1st_sem_approved": [0, 3]})
    out = AcademicFeatureEngineer(semesters=("1st",)).fit_transform(X)
    assert out["curricular_units_1st_sem_approval_rate"].tolist() == [0.0, 0.75]


@pytest.mark.parametrize("kind", ["logreg", "rf", "hgb"])
def test_preprocessor_unseen_categories(clean, kind):
    X = clean[FULL.raw_columns]
    eng = AcademicFeatureEngineer().fit(X)
    prep = build_preprocessor(kind, FULL).fit(eng.transform(X))
    new = X.head(3).copy()
    new["course"] = 99999
    new["fathers_occupation"] = 777
    Z = prep.transform(eng.transform(new))
    assert Z.shape[0] == 3
    if kind != "hgb":
        assert np.isfinite(Z).all()


def test_scaler_fitted_only_on_given_data(clean):
    """El escalado se aprende del subconjunto con que se ajusta (no del dataset completo)."""
    X = AcademicFeatureEngineer().fit_transform(clean[FULL.raw_columns])
    sub = X.iloc[:100]
    prep = build_preprocessor("logreg", FULL).fit(sub)
    scaler = prep.named_transformers_["numeric"]
    assert np.allclose(scaler.mean_[FULL.numeric_with_engineered.index("admission_grade")],
                       sub["admission_grade"].mean())


def test_validate_input_ok_and_errors(clean):
    row = clean[FULL.raw_columns].head(1)
    out, warns = validate_input(row.assign(extra=1), FULL)
    assert list(out.columns) == FULL.raw_columns and warns
    with pytest.raises(SchemaError):
        validate_input(row.drop(columns=["course"]), FULL)
    with pytest.raises(SchemaError):
        validate_input(row.assign(curricular_units_1st_sem_grade=25.0), FULL)
    with pytest.raises(SchemaError):
        validate_input(row.assign(debtor=2), FULL)
    with pytest.raises(SchemaError):
        validate_input(row.assign(curricular_units_1st_sem_enrolled=2, curricular_units_1st_sem_approved=5), FULL)
    with pytest.raises(SchemaError):
        validate_input(row.assign(admission_grade="alto"), FULL)
    _, warns = validate_input(row.assign(course=12345), FULL)
    assert any("no documentado" in w for w in warns)


def test_every_observed_code_is_documented(clean):
    for col in dd.NOMINAL_FEATURES:
        undocumented = set(clean[col]) - set(dd.CATEGORY_CODES[col])
        assert not undocumented, f"{col}: {undocumented}"


def test_column_names_match_draft_notebook_normalizer(raw):
    """El esquema del pipeline es el mismo que produce `normalizar_nombre` del notebook borrador."""
    assert [normalizar_nombre(c) for c in raw.columns] == list(dd.COLUMN_MAP.values())


def test_soft_rules_are_flagged_not_removed(clean):
    flags = soft_rule_flags(clean)
    assert int(flags.sum()) == 912          # mismo conteo que reporta el EDA del borrador
    assert len(clean) == 4424               # ninguna fila se elimina


def test_winsorizer_learns_limits_only_from_fit_data(clean):
    X = clean[FULL.raw_columns]
    w = WinsorizadorIQR().fit(X.iloc[:1000])
    q1, q3 = X["age_at_enrollment"].iloc[:1000].quantile([0.25, 0.75])
    assert w.limites_["age_at_enrollment"][1] == pytest.approx(q3 + 1.5 * (q3 - q1))
    out = w.transform(X)
    assert out["age_at_enrollment"].max() <= w.limites_["age_at_enrollment"][1]
    assert X["age_at_enrollment"].max() == 70   # el original no se modifica
