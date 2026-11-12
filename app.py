"""EduPredict AI — aplicación Streamlit.

Ejecutar con::

    streamlit run app.py

La aplicación NO entrena modelos: carga los artefactos de ``models/`` generados
por el notebook ``notebooks/ACA2_Student_Dropout_NN.ipynb``. Todas las
predicciones provienen de una red neuronal (PyTorch): MLP o FT-Transformer.
Los datos se leen de la base SQLite ``data/estudiantes.db``.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import data_dictionary as dd
from src import project_info as info
from src.data_loader import decode_for_display, load_clean
from src.db import DB_PATH, load_clean_from_db
from src.predict import (
    ModelNotAvailableError,
    check_artifacts,
    describe_prediction,
    feature_set,
    load_metadata,
    load_metrics,
    predict_one,
)
from src.preprocessing import SchemaError

# ---------------------------------------------------------------------------
# Configuración e identidad visual
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="auto",
)

NAVY = "#0B2545"
NAVY_2 = "#13315C"
MUTED = "#5B6577"
GRID = "#E6E9EF"
CLASS_ORDER = dd.CLASSES
COLOR_MAP_ES = {dd.CLASS_LABELS_ES[c]: dd.CLASS_COLORS[c] for c in CLASS_ORDER}
ORDER_ES = [dd.CLASS_LABELS_ES[c] for c in CLASS_ORDER]

# CSS mínimo: solo clases propias (sin depender de selectores internos de Streamlit).
st.markdown(
    """
    <style>
    .ep-hero {background: linear-gradient(120deg, #0B2545 0%, #13315C 100%);
              color: #FFFFFF; padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}
    .ep-hero h1 {color: #FFFFFF; font-size: 1.9rem; margin: 0 0 .25rem 0; font-weight: 700;}
    .ep-hero p {color: #DCE3EE; margin: 0; font-size: 1rem;}
    .ep-badge {display: inline-block; padding: .18rem .6rem; border-radius: 999px;
               font-size: .75rem; font-weight: 600; letter-spacing: .02em; margin-bottom: .5rem;}
    .ep-badge-hist {background: #E8EEF7; color: #13315C;}
    .ep-badge-model {background: #FFF4DE; color: #7A4F00;}
    .ep-kpi {background: #FFFFFF; border: 1px solid #E6E9EF; border-radius: 12px;
             padding: 1rem 1.1rem; height: 100%;}
    .ep-kpi .ep-kpi-label {color: #5B6577; font-size: .85rem; margin-bottom: .2rem;}
    .ep-kpi .ep-kpi-value {color: #0B2545; font-size: 1.8rem; font-weight: 700; line-height: 1.1;}
    .ep-kpi .ep-kpi-sub {color: #5B6577; font-size: .78rem; margin-top: .2rem;}
    .ep-kpi .ep-dot {display:inline-block; width:.65rem; height:.65rem; border-radius:50%;
                     margin-right:.4rem; vertical-align: middle;}
    .ep-result {border-radius: 14px; padding: 1.1rem 1.3rem; color: #FFFFFF;}
    .ep-result .ep-result-label {font-size: .85rem; opacity: .9;}
    .ep-result .ep-result-value {font-size: 1.7rem; font-weight: 700;}
    .ep-section-title {color: #0B2545; font-weight: 700; font-size: 1.05rem; margin: .4rem 0 .2rem 0;}
    .ep-muted {color: #5B6577; font-size: .88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="ep-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "hist") -> None:
    st.markdown(f'<span class="ep-badge ep-badge-{kind}">{text}</span>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str | None = None) -> str:
    dot = f'<span class="ep-dot" style="background:{color}"></span>' if color else ""
    return (
        f'<div class="ep-kpi"><div class="ep-kpi-label">{dot}{label}</div>'
        f'<div class="ep-kpi-value">{value}</div><div class="ep-kpi-sub">{sub}</div></div>'
    )


def style_fig(fig: go.Figure, height: int = 360, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        separators=",.",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", color="#1B2433", size=13),
        title_font=dict(size=15, color=NAVY),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, title=None),
        bargap=0.25,
        hoverlabel=dict(bgcolor="#FFFFFF", font_size=13),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ---------------------------------------------------------------------------
# Datos y artefactos (cacheados)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    # Paso 1 del ACA 2: los datos se consultan desde SQLite; si la base no está, se usa el CSV limpio.
    try:
        df = load_clean_from_db() if DB_PATH.exists() else load_clean()
    except Exception:  # base corrupta o esquema distinto -> respaldo
        df = load_clean()
    df["curricular_units_1st_sem_approval_rate"] = (df["curricular_units_1st_sem_approved"] / df["curricular_units_1st_sem_enrolled"].where(df["curricular_units_1st_sem_enrolled"] > 0)).fillna(0)
    df["curricular_units_2nd_sem_approval_rate"] = (df["curricular_units_2nd_sem_approved"] / df["curricular_units_2nd_sem_enrolled"].where(df["curricular_units_2nd_sem_enrolled"] > 0)).fillna(0)
    return df


@st.cache_data(show_spinner=False)
def get_display_data() -> pd.DataFrame:
    return decode_for_display(get_data())


@st.cache_resource(show_spinner=False)
def get_artifacts(modality: str):
    return load_metadata(modality), load_metrics(modality)


def pct(x: float) -> str:
    return f"{x:.1%}".replace(".", ",")


def num(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}".replace(".", ",")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## 🎓 {info.NOMBRE}")
    st.caption("Predicción de deserción universitaria con redes neuronales (MLP · FT-Transformer)")
    section = st.radio(
        "Navegación",
        ["Dashboard", "Predicción individual", "Comparación de modelos", "Acerca del proyecto"],
        label_visibility="collapsed",
    )
    st.divider()
    missing = check_artifacts()
    if any(missing.values()):
        st.warning("Faltan modelos entrenados. Ejecute el notebook del ACA 2 para generar `models/*.pt`.")
    else:
        st.success("Modelos cargados desde `models/`", icon="✅")
    st.caption("Dataset: UCI 697 — *Predict Students' Dropout and Academic Success*")
    st.caption("Herramienta académica. No sustituye decisiones institucionales.")


# ---------------------------------------------------------------------------
# Sección A — Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    hero("Dashboard académico", "Indicadores descriptivos del histórico de 4.424 estudiantes (UCI 697).")
    badge("DATOS HISTÓRICOS · no son predicciones del modelo", "hist")

    disp = get_display_data()
    raw = get_data()

    # Filtros en una sola fila, encima de los gráficos
    f1, f2, f3, f4 = st.columns([2.2, 1, 1, 1])
    cursos = sorted(disp["course"].unique())
    sel_cursos = f1.multiselect("Programa académico", cursos, placeholder="Todos los programas")
    sel_jornada = f2.selectbox("Jornada", ["Todas", "Diurna", "Nocturna"])
    sel_sexo = f3.selectbox("Sexo", ["Todos", "Femenino", "Masculino"])
    sel_beca = f4.selectbox("Becario", ["Todos", "Sí", "No"])

    mask = pd.Series(True, index=disp.index)
    if sel_cursos:
        mask &= disp["course"].isin(sel_cursos)
    if sel_jornada != "Todas":
        mask &= disp["daytime_evening_attendance"] == sel_jornada
    if sel_sexo != "Todos":
        mask &= disp["gender"] == sel_sexo
    if sel_beca != "Todos":
        mask &= disp["scholarship_holder"] == sel_beca
    d, r = disp[mask], raw[mask]

    if d.empty:
        st.info("No hay estudiantes con la combinación de filtros seleccionada.")
        return

    n = len(d)
    shares = d[dd.TARGET].value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0)
    counts = d[dd.TARGET].value_counts().reindex(CLASS_ORDER, fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Total de estudiantes", f"{n:,}".replace(",", "."),
                    "registros en la selección" if mask.sum() < len(disp) else "dataset completo"), unsafe_allow_html=True)
    c2.markdown(kpi("Deserción histórica", pct(shares["Dropout"]), f"{counts['Dropout']} estudiantes",
                    dd.CLASS_COLORS["Dropout"]), unsafe_allow_html=True)
    c3.markdown(kpi("Matriculados", pct(shares["Enrolled"]), f"{counts['Enrolled']} estudiantes",
                    dd.CLASS_COLORS["Enrolled"]), unsafe_allow_html=True)
    c4.markdown(kpi("Graduados", pct(shares["Graduate"]), f"{counts['Graduate']} estudiantes",
                    dd.CLASS_COLORS["Graduate"]), unsafe_allow_html=True)
    st.write("")

    col_a, col_b = st.columns([1, 1.6])
    with col_a:
        dist = pd.DataFrame({"clase": ORDER_ES, "n": counts.values, "p": shares.values})
        fig = go.Figure(go.Pie(
            labels=dist["clase"], values=dist["n"], hole=0.62, sort=False,
            marker=dict(colors=[COLOR_MAP_ES[c] for c in dist["clase"]], line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent", textposition="outside",
            hovertemplate="%{label}: %{value} estudiantes (%{percent})<extra></extra>",
        ))
        fig.update_layout(title="Distribución de la situación final",
                          annotations=[dict(text=f"<b>{n:,}</b><br>estudiantes".replace(",", "."),
                                            showarrow=False, font=dict(size=14, color=NAVY))])
        st.plotly_chart(style_fig(fig, 380, legend=False), width="stretch", config=PLOT_CONFIG)
    with col_b:
        by_course = (d.groupby("course")[dd.TARGET].apply(lambda s: (s == "Dropout").mean())
                     .rename("tasa").reset_index().merge(d["course"].value_counts().rename("n").reset_index(), on="course"))
        by_course = by_course.sort_values("tasa")
        fig = px.bar(by_course, x="tasa", y="course", orientation="h", custom_data=["n"],
                     text=by_course["tasa"].map(pct))
        fig.update_traces(marker_color=dd.CLASS_COLORS["Dropout"], textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>Deserción: %{x:.1%}<br>n = %{customdata[0]}<extra></extra>")
        fig.update_layout(title="Tasa de deserción histórica por programa", xaxis_title=None, yaxis_title=None)
        fig.update_xaxes(tickformat=".0%", range=[0, min(1, by_course["tasa"].max() * 1.25 + 0.02)])
        st.plotly_chart(style_fig(fig, max(380, 26 * len(by_course) + 80), legend=False),
                        width="stretch", config=PLOT_CONFIG)

    st.markdown('<div class="ep-section-title">Rendimiento académico según la situación final</div>', unsafe_allow_html=True)
    perf_vars = {
        dd.label("curricular_units_1st_sem_grade"): "curricular_units_1st_sem_grade",
        dd.label("curricular_units_2nd_sem_grade"): "curricular_units_2nd_sem_grade",
        dd.label("curricular_units_1st_sem_approval_rate"): "curricular_units_1st_sem_approval_rate",
        dd.label("curricular_units_2nd_sem_approval_rate"): "curricular_units_2nd_sem_approval_rate",
        dd.label("curricular_units_1st_sem_approved"): "curricular_units_1st_sem_approved",
        dd.label("admission_grade"): "admission_grade",
        dd.label("age_at_enrollment"): "age_at_enrollment",
    }
    col_c, col_d = st.columns([1.3, 1])
    with col_c:
        var_lbl = st.selectbox("Variable", list(perf_vars), label_visibility="collapsed")
        var = perf_vars[var_lbl]
        tmp = r[[var, dd.TARGET]].assign(clase=r[dd.TARGET].map(dd.CLASS_LABELS_ES))
        fig = px.box(tmp, x="clase", y=var, color="clase", category_orders={"clase": ORDER_ES},
                     color_discrete_map=COLOR_MAP_ES, points=False)
        fig.update_layout(title=f"{var_lbl} por situación final", xaxis_title=None,
                          yaxis_title=dd.FEATURE_INFO.get(var, ("", ""))[1])
        if "rate" in var:
            fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_fig(fig, 380, legend=False), width="stretch", config=PLOT_CONFIG)
    with col_d:
        means = (r.groupby(dd.TARGET)[["curricular_units_1st_sem_grade", "curricular_units_2nd_sem_grade", "curricular_units_1st_sem_approval_rate",
                                       "curricular_units_2nd_sem_approval_rate", "admission_grade", "age_at_enrollment"]]
                 .mean().reindex(CLASS_ORDER).T)
        means.columns = ORDER_ES
        means.index = [dd.label(i) for i in means.index]
        st.markdown('<div class="ep-muted">Promedios por situación final</div>', unsafe_allow_html=True)
        st.dataframe(means.style.format(lambda v: num(v, 2)), width="stretch")

    st.markdown('<div class="ep-section-title">Variables asociadas con la deserción</div>', unsafe_allow_html=True)
    st.caption("Tasa de deserción observada según cada condición. Muestra asociación descriptiva, no causalidad.")
    rows = []
    for col in ["tuition_fees_up_to_date", "debtor", "scholarship_holder", "displaced", "gender",
                "daytime_evening_attendance", "international", "educational_special_needs"]:
        for val in (1, 0):
            sub = r[r[col] == val]
            if len(sub):
                rows.append({"variable": dd.label(col), "condicion": dd.code_label(col, val),
                             "grupo": "Sí / Diurna / Masculino" if val == 1 else "No / Nocturna / Femenino",
                             "tasa": (sub[dd.TARGET] == "Dropout").mean(), "n": len(sub)})
    assoc = pd.DataFrame(rows)
    if not assoc.empty:
        order = (assoc.pivot_table(index="variable", columns="grupo", values="tasa")
                 .diff(axis=1).abs().iloc[:, -1].sort_values().index.tolist())
        fig = px.bar(assoc, x="tasa", y="variable", color="grupo", orientation="h", barmode="group",
                     category_orders={"variable": order[::-1]}, custom_data=["condicion", "n"],
                     text=assoc["tasa"].map(pct),
                     color_discrete_sequence=[NAVY_2, "#9AA7BD"])
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{y} = %{customdata[0]}<br>Deserción: %{x:.1%}<br>n = %{customdata[1]}<extra></extra>")
        fig.update_xaxes(tickformat=".0%", range=[0, min(1, assoc["tasa"].max() * 1.2 + 0.02)])
        fig.update_layout(xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style_fig(fig, 460), width="stretch", config=PLOT_CONFIG)

    with st.expander("Correlaciones entre variables numéricas"):
        num_cols = [c for c in dd.NUMERIC_FEATURES if c in r]
        corr = r[num_cols].corr(method="spearman")
        labels = [dd.label(c) for c in num_cols]
        fig = go.Figure(go.Heatmap(z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, zmid=0,
                                   colorscale="RdBu", reversescale=True,
                                   hovertemplate="%{y}<br>%{x}<br>ρ = %{z:.2f}<extra></extra>"))
        fig.update_layout(title="Correlación de Spearman")
        st.plotly_chart(style_fig(fig, 640, legend=False), width="stretch", config=PLOT_CONFIG)
        st.caption("Se usa Spearman porque varias variables son conteos discretos con distribuciones asimétricas. "
                   "Las variables de 1.er y 2.º semestre están fuertemente correlacionadas entre sí.")

    with st.expander("Ver datos filtrados"):
        st.dataframe(d.drop(columns=[dd.TARGET]).head(500), width="stretch", hide_index=True)
        st.caption(f"Mostrando hasta 500 de {len(d)} registros.")


# ---------------------------------------------------------------------------
# Sección B — Predicción individual
# ---------------------------------------------------------------------------
def _select(container, col: str, meta: dict, freq: pd.Series, help_txt: str | None = None):
    """Selector descriptivo: muestra el texto en español y devuelve el código interno."""
    codes = sorted(dd.CATEGORY_CODES[col], key=lambda c: (-freq.get(c, 0), c)) if col in dd.CATEGORY_CODES else [1, 0]
    default = meta["modas_entrenamiento"].get(col, codes[0])
    idx = codes.index(default) if default in codes else 0
    return container.selectbox(dd.label(col), codes, index=idx, help=help_txt,
                               format_func=lambda c: dd.code_label(col, c), key=f"in_{col}")


def _number(container, col: str, meta: dict, step=1, fmt=None, help_txt: str | None = None):
    lo, hi = dd.VALID_RANGES[col]
    stats = meta["estadisticas_numericas_entrenamiento"][col]
    is_int = isinstance(step, int)
    default = stats["mediana"]
    default = int(round(default)) if is_int else float(default)
    unit = dd.FEATURE_INFO[col][1]
    return container.number_input(
        f"{dd.label(col)}", min_value=int(lo) if is_int else float(lo),
        max_value=int(hi) if is_int else float(hi), value=default, step=step, format=fmt,
        help=help_txt or f"Unidad: {unit}. Rango observado en entrenamiento: {stats['min']}–{stats['max']}.",
        key=f"in_{col}",
    )


def page_predict() -> None:
    hero("Predicción individual", "Estimación de la situación final de un estudiante con el modelo entrenado.")
    badge("PREDICCIÓN DEL MODELO · herramienta académica", "model")

    missing = check_artifacts()
    modalidad_lbl = st.radio(
        "Modalidad de predicción",
        ["Información académica completa", "Alerta temprana (cierre del 1.er semestre)"],
        horizontal=True,
        help="Son modelos distintos entrenados con conjuntos de variables diferentes; sus resultados no son directamente comparables.",
    )
    modality = "completo" if modalidad_lbl.startswith("Información") else "temprano"
    if missing[modality]:
        st.error(f"No se encontraron los artefactos del modelo {modality}: {missing[modality]}. "
                 "Ejecute el notebook del ACA 2 para generarlos.")
        return
    meta, metrics = get_artifacts(modality)
    fs = feature_set(modality)
    st.caption(f"{fs.description} Modelo: **{meta['algoritmo_nombre']}**.")

    data = get_data()
    freqs = {c: data[c].value_counts() for c in dd.NOMINAL_FEATURES}
    macro = (data.groupby(["unemployment_rate", "inflation_rate", "gdp"]).size()
             .rename("n").reset_index().sort_values("n", ascending=False))
    macro_opts = [(float(u), float(i), float(g)) for u, i, g, _ in macro.itertuples(index=False, name=None)]

    with st.form(f"form_{modality}"):
        t_ing, t_per, t_soc, t_s1, t_s2 = st.tabs(
            ["🏛️ Ingreso y programa", "👤 Datos personales", "💼 Socioeconómico",
             "📘 1.er semestre", "📗 2.º semestre"]
        )
        rec: dict = {}
        with t_ing:
            a, b = st.columns(2)
            rec["course"] = _select(a, "course", meta, freqs["course"])
            rec["daytime_evening_attendance"] = b.selectbox(dd.label("daytime_evening_attendance"), [1, 0],
                                                    format_func=lambda v: dd.code_label("daytime_evening_attendance", v),
                                                    key="in_daytime_attendance")
            rec["application_mode"] = _select(a, "application_mode", meta, freqs["application_mode"])
            rec["application_order"] = b.slider(dd.label("application_order"), 0, 9,
                                                int(meta["estadisticas_numericas_entrenamiento"]["application_order"]["mediana"]),
                                                help="0 = primera opción, 9 = última opción", key="in_application_order")
            rec["previous_qualification"] = _select(a, "previous_qualification", meta, freqs["previous_qualification"])
            rec["previous_qualification_grade"] = _number(b, "previous_qualification_grade", meta, step=0.5, fmt="%.1f")
            rec["admission_grade"] = _number(b, "admission_grade", meta, step=0.5, fmt="%.1f")
            macro_sel = a.selectbox(
                "Contexto macroeconómico del año de ingreso",
                macro_opts,
                format_func=lambda t: f"Desempleo {num(t[0], 1)} % · Inflación {num(t[1], 1)} % · PIB {num(t[2], 2)}",
                help="El dataset solo contiene 10 combinaciones reales (una por cohorte). Se ofrecen esas combinaciones.",
                key="in_macro",
            )
            rec["unemployment_rate"], rec["inflation_rate"], rec["gdp"] = map(float, macro_sel)
        with t_per:
            a, b = st.columns(2)
            rec["age_at_enrollment"] = _number(a, "age_at_enrollment", meta, step=1)
            rec["gender"] = b.selectbox(dd.label("gender"), [0, 1], format_func=lambda v: dd.code_label("gender", v),
                                        key="in_gender")
            rec["marital_status"] = _select(a, "marital_status", meta, freqs["marital_status"])
            rec["nationality"] = _select(b, "nationality", meta, freqs["nationality"])
            yn = lambda v: dd.YES_NO[v]  # noqa: E731
            rec["international"] = a.selectbox(dd.label("international"), [0, 1], format_func=yn, key="in_international")
            rec["displaced"] = b.selectbox(dd.label("displaced"), [1, 0], format_func=yn, key="in_displaced",
                                           index=0 if meta["modas_entrenamiento"]["displaced"] == 1 else 1)
            rec["educational_special_needs"] = a.selectbox(dd.label("educational_special_needs"), [0, 1], format_func=yn, key="in_special_needs")
        with t_soc:
            a, b = st.columns(2)
            rec["scholarship_holder"] = a.selectbox(dd.label("scholarship_holder"), [0, 1], format_func=yn, key="in_scholarship_holder")
            rec["tuition_fees_up_to_date"] = b.selectbox(dd.label("tuition_fees_up_to_date"), [1, 0], format_func=yn, key="in_tuition_up_to_date")
            rec["debtor"] = a.selectbox(dd.label("debtor"), [0, 1], format_func=yn, key="in_debtor")
            rec["mothers_qualification"] = _select(a, "mothers_qualification", meta, freqs["mothers_qualification"])
            rec["fathers_qualification"] = _select(b, "fathers_qualification", meta, freqs["fathers_qualification"])
            rec["mothers_occupation"] = _select(a, "mothers_occupation", meta, freqs["mothers_occupation"])
            rec["fathers_occupation"] = _select(b, "fathers_occupation", meta, freqs["fathers_occupation"])

        def semester_inputs(tab, sem: str):
            a, b, c = tab.columns(3)
            out = {
                f"curricular_units_{sem}_sem_enrolled": _number(a, f"curricular_units_{sem}_sem_enrolled", meta),
                f"curricular_units_{sem}_sem_approved": _number(b, f"curricular_units_{sem}_sem_approved", meta),
                f"curricular_units_{sem}_sem_grade": _number(c, f"curricular_units_{sem}_sem_grade", meta, step=0.1, fmt="%.2f",
                                           help_txt="Promedio en escala 0–20. Use 0 si no aprobó ninguna unidad."),
                f"curricular_units_{sem}_sem_evaluations": _number(a, f"curricular_units_{sem}_sem_evaluations", meta),
                f"curricular_units_{sem}_sem_credited": _number(b, f"curricular_units_{sem}_sem_credited", meta),
                f"curricular_units_{sem}_sem_without_evaluations": _number(c, f"curricular_units_{sem}_sem_without_evaluations", meta),
            }
            return out

        with t_s1:
            rec.update(semester_inputs(t_s1, "1st"))
        with t_s2:
            if modality == "completo":
                rec.update(semester_inputs(t_s2, "2nd"))
            else:
                st.info("La alerta temprana no utiliza información del 2.º semestre: en el momento de la "
                        "predicción esos resultados aún no existen.")

        submitted = st.form_submit_button("Analizar estudiante", type="primary", width="stretch")

    if submitted:
        try:
            res = predict_one(rec, modality)
            st.session_state["last_pred"] = {"modality": modality, "res": res, "rec": rec}
        except SchemaError as exc:
            st.session_state.pop("last_pred", None)
            st.error(f"Datos no válidos: {exc}")
        except ModelNotAvailableError as exc:
            st.error(str(exc))

    last = st.session_state.get("last_pred")
    if not last or last["modality"] != modality:
        st.caption("Complete el formulario y presione **Analizar estudiante**. Los valores iniciales son "
                   "las medianas/modas del conjunto de entrenamiento.")
        return

    res = last["res"]
    for w in res.advertencias:
        st.warning(w)
    st.markdown("---")
    left, right = st.columns([1, 1.5])
    with left:
        color = dd.CLASS_COLORS[res.clase]
        st.markdown(
            f'<div class="ep-result" style="background:{color}">'
            f'<div class="ep-result-label">Clasificación estimada · {"modelo completo" if modality == "completo" else "alerta temprana"}</div>'
            f'<div class="ep-result-value">{res.clase_es}</div>'
            f'<div class="ep-result-label">Probabilidad estimada: {pct(res.probabilidades[res.clase])}</div></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(describe_prediction(res))
    with right:
        pf = res.as_frame()
        fig = px.bar(pf, x="probabilidad", y="clase_es", orientation="h", color="clase_es",
                     color_discrete_map=COLOR_MAP_ES, text=pf["probabilidad"].map(pct),
                     category_orders={"clase_es": ORDER_ES[::-1]})
        fig.update_traces(textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}: %{x:.1%}<extra></extra>")
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title=None)
        fig.update_yaxes(title=None)
        fig.update_layout(title="Probabilidades estimadas por clase")
        st.plotly_chart(style_fig(fig, 260, legend=False), width="stretch", config=PLOT_CONFIG)

    st.info(
        "**Herramienta académica.** Las probabilidades son estimaciones de un modelo estadístico entrenado con datos "
        "históricos de una institución portuguesa; no constituyen una medición validada de riesgo ni una decisión "
        "definitiva sobre el estudiante. Úselas solo como apoyo para la reflexión y el seguimiento académico.",
        icon="ℹ️",
    )
    with st.expander("Ver datos enviados al modelo"):
        shown = pd.DataFrame({
            "Variable": [dd.label(k) for k in last["rec"]],
            "Valor descriptivo": [dd.code_label(k, v) if (k in dd.NOMINAL_FEATURES or k in dd.BINARY_FEATURES) else v
                                  for k, v in last["rec"].items()],
            "Valor interno": [str(v) for v in last["rec"].values()],
        })
        st.dataframe(shown, hide_index=True, width="stretch")


# ---------------------------------------------------------------------------
# Sección C — Comparación de modelos
# ---------------------------------------------------------------------------
NN_KEYS = ("mlp", "ft_transformer")


def render_curves(h: dict, title_suffix: str = "") -> None:
    """Curvas de entrenamiento / validación (pérdida y macro-F1) con la época elegida."""
    epochs = list(range(1, len(h["val_f1"]) + 1))
    c1, c2 = st.columns(2)
    for col, (tr, va, name, fmt) in zip((c1, c2), (("train_loss", "val_loss", "Pérdida (entropía cruzada ponderada)", ".3f"),
                                                    ("train_f1", "val_f1", "Macro-F1", ".3f"))):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=h[tr], name="Entrenamiento", line=dict(color=NAVY_2, width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=h[va], name="Validación", line=dict(color="#eb6834", width=2)))
        fig.add_vline(x=h["best_epoch"], line_dash="dot", line_color=MUTED,
                      annotation_text=f"época elegida: {h['best_epoch']}", annotation_position="top right")
        fig.update_layout(title=f"{name}{title_suffix}", xaxis_title="Época")
        with col:
            st.plotly_chart(style_fig(fig, 320), width="stretch", config=PLOT_CONFIG)


def render_comparison(modality: str) -> None:
    meta, m = get_artifacts(modality)
    models = m["modelos"]
    winner = m["modelo_seleccionado"]
    classic = [k for k in models if k not in NN_KEYS]

    def label(k: str) -> str:
        return models[k]["nombre"] + ("" if k in NN_KEYS else " · ACA 1")

    rows = []
    for k, v in models.items():
        rows.append({
            "Algoritmo": ("★ " if k == winner else "") + label(k),
            "Macro-F1 (CV)": f"{num(v['cv']['f1_macro']['media'])} ± {num(v['cv']['f1_macro']['desv'])}",
            "Accuracy": v["prueba"]["accuracy"],
            "Macro-F1": v["prueba"]["f1_macro"],
            "Balanced acc.": v["prueba"]["balanced_accuracy"],
            "Recall Deserción": v["prueba"]["por_clase"]["Dropout"]["recall"],
            "Recall Matriculado": v["prueba"]["por_clase"]["Enrolled"]["recall"],
            "Recall Graduado": v["prueba"]["por_clase"]["Graduate"]["recall"],
            "ROC-AUC (OvR)": v["prueba"].get("roc_auc_ovr_macro"),
        })
    table = pd.DataFrame(rows)
    st.markdown('<div class="ep-section-title">Tabla comparativa</div>', unsafe_allow_html=True)
    st.caption("★ = red neuronal desplegada en la app. «Macro-F1 (CV)»: media ± desviación en validación cruzada "
               "estratificada de 5 pliegues (criterio de selección). El resto de columnas: evaluación final en el "
               f"conjunto de prueba (20 %, n = {meta['particion']['n_prueba']}). Los modelos «ACA 1» son los clásicos "
               "de la entrega anterior, con la misma partición y validación, incluidos como referencia.")
    st.dataframe(table.style.format({c: (lambda v: num(v)) for c in table.columns[2:]}),
                 hide_index=True, width="stretch")

    w = models[winner]
    others_nn = [k for k in NN_KEYS if k in models and k != winner]
    best_classic = max(classic, key=lambda k: models[k]["cv"]["f1_macro"]["media"]) if classic else None
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.markdown('<div class="ep-section-title">Red neuronal desplegada</div>', unsafe_allow_html=True)
        st.markdown(kpi(w["nombre"], num(w["cv"]["f1_macro"]["media"]), "macro-F1 medio en validación cruzada",
                        NAVY), unsafe_allow_html=True)
        txt = ("**Justificación.** Entre las redes neuronales evaluadas obtuvo el mayor macro-F1 en validación cruzada "
               f"({num(w['cv']['f1_macro']['media'])})")
        if others_nn:
            o = models[others_nn[0]]
            txt += (f", frente a {num(o['cv']['f1_macro']['media'])} de {o['nombre']}")
        txt += (". Se usa macro-F1 porque pondera por igual las tres clases y no premia ignorar la clase minoritaria "
                "«Matriculado». La elección se hizo **sin** consultar el conjunto de prueba.")
        st.markdown(txt)
        if best_classic:
            bc = models[best_classic]
            diff = w["cv"]["f1_macro"]["media"] - bc["cv"]["f1_macro"]["media"]
            rel = "supera a" if diff > 0 else "queda por debajo de"
            st.caption(f"Frente al mejor modelo clásico del ACA 1 ({bc['nombre']}, {num(bc['cv']['f1_macro']['media'])}), "
                       f"la red {rel} ese valor por {num(abs(diff))}. "
                       + ("La diferencia es menor que la desviación entre pliegues: el desempeño es equivalente. "
                          if abs(diff) < max(w['cv']['f1_macro']['desv'], bc['cv']['f1_macro']['desv']) else ""))
        st.markdown("**Hiperparámetros elegidos (validación cruzada)**")
        st.json(w["mejores_hiperparametros"], expanded=True)
        if meta.get("parametros_entrenables"):
            st.caption(f"Parámetros entrenables: {meta['parametros_entrenables']:,}".replace(",", "."))
    with c2:
        keys = list(models)
        sel = st.selectbox("Matriz de confusión del modelo", keys, index=keys.index(winner),
                           format_func=label, key=f"cm_{modality}")
        cm = models[sel]["prueba"]["matriz_confusion"]
        lbl = [dd.CLASS_LABELS_ES[c] for c in cm["etiquetas"]]
        z = pd.DataFrame(cm["valores"], index=lbl, columns=lbl)
        zn = z.div(z.sum(axis=1), axis=0)
        text = [[f"{z.iat[i, j]}<br>({pct(zn.iat[i, j])})" for j in range(3)] for i in range(3)]
        fig = go.Figure(go.Heatmap(z=zn.values, x=lbl, y=lbl, text=text, texttemplate="%{text}",
                                   colorscale=[[0, "#F4F6FA"], [1, NAVY_2]], zmin=0, zmax=1, showscale=False,
                                   hovertemplate="Real: %{y}<br>Predicho: %{x}<br>%{text}<extra></extra>"))
        fig.update_layout(title=f"Matriz de confusión (prueba) — {models[sel]['nombre']}",
                          xaxis_title="Clase predicha", yaxis_title="Clase real")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 380, legend=False), width="stretch", config=PLOT_CONFIG)
        st.caption("Porcentajes por fila: proporción de cada clase real que el modelo asignó a cada clase (recall en la diagonal).")

    if w.get("historial"):
        st.markdown('<div class="ep-section-title">Curvas de entrenamiento de la red desplegada</div>',
                    unsafe_allow_html=True)
        render_curves(w["historial"])
        st.caption("Entrenamiento y validación se miden al final de cada época en modo evaluación (sin dropout). "
                   "Se conservan los pesos de la época con mejor macro-F1 de validación (early stopping).")

    c3, c4 = st.columns(2)
    with c3:
        per = []
        for k, v in models.items():
            for cls, vals in v["prueba"]["por_clase"].items():
                per.append({"Algoritmo": label(k).split(" (")[0].replace(" · ACA 1", " (ACA 1)"), "Clase": dd.CLASS_LABELS_ES[cls],
                            "Precisión": vals["precision"], "Recall": vals["recall"], "F1": vals["f1"]})
        per = pd.DataFrame(per)
        metric = st.radio("Métrica por clase", ["F1", "Recall", "Precisión"], horizontal=True, key=f"pc_{modality}")
        fig = px.bar(per, x="Clase", y=metric, color="Algoritmo", barmode="group",
                     category_orders={"Clase": ORDER_ES}, text=per[metric].map(lambda v: num(v, 2)),
                     color_discrete_sequence=["#13315C", "#2a78d6", "#eb6834", "#1baf7a", "#8E8EA0"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(range=[0, 1.08], title=None)
        fig.update_layout(title=f"{metric} por clase en prueba", xaxis_title=None)
        st.plotly_chart(style_fig(fig, 380), width="stretch", config=PLOT_CONFIG)
    with c4:
        imp = pd.DataFrame(m["importancia_permutacion_prueba"]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["importancia_media"], y=imp["etiqueta"], orientation="h",
                               error_x=dict(type="data", array=imp["importancia_desv"], color=MUTED, thickness=1),
                               marker_color=NAVY_2,
                               hovertemplate="%{y}<br>Caída de macro-F1: %{x:.3f}<extra></extra>"))
        fig.update_layout(title=f"Importancia por permutación — {w['nombre'].split(' (')[0]}",
                          xaxis_title="Caída media de macro-F1")
        st.plotly_chart(style_fig(fig, 420, legend=False), width="stretch", config=PLOT_CONFIG)
        st.caption("Calculada sobre el conjunto de prueba con 10 repeticiones. Indica cuánto depende el modelo de "
                   "cada variable; no implica causalidad.")


def page_compare() -> None:
    hero("Comparación de modelos", "Redes neuronales (MLP · FT-Transformer) frente a los modelos clásicos del ACA 1")
    badge("MÉTRICAS REALES · cargadas desde models/*.json", "hist")
    missing = check_artifacts()
    tabs = st.tabs(["Modelo con información completa", "Modelo de alerta temprana"])
    for tab, modality in zip(tabs, ["completo", "temprano"]):
        with tab:
            if missing[modality]:
                st.error(f"Faltan artefactos: {missing[modality]}. Ejecute el notebook del ACA 2.")
                continue
            st.caption(feature_set(modality).description)
            render_comparison(modality)
    st.info("Los dos modelos usan conjuntos de información distintos: el de alerta temprana renuncia a los datos del "
            "2.º semestre para poder anticipar la deserción antes. Sus métricas describen escenarios diferentes y no "
            "deben compararse como si fueran el mismo problema.", icon="ℹ️")


# ---------------------------------------------------------------------------
# Sección D — Acerca del proyecto
# ---------------------------------------------------------------------------
def page_about() -> None:
    hero("Acerca del proyecto", info.DESCRIPCION)
    st.markdown(f"**Contexto académico.** {info.CONTEXTO}")
    st.markdown(f"**Pilar:** {info.PILAR} · **Docente:** {info.DOCENTE}")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 🎯 Objetivo general")
            st.write(info.OBJETIVO)
    with c2:
        with st.container(border=True):
            st.markdown("#### ❓ Pregunta orientadora")
            st.write(info.PREGUNTA)

    st.markdown("#### 📊 Dataset")
    df = get_data()
    st.markdown(
        f"*Predict Students' Dropout and Academic Success* — UCI Machine Learning Repository (ID 697). "
        f"{len(df):,} registros".replace(",", ".") + ", 36 variables predictoras (académicas, socioeconómicas, "
        "demográficas y macroeconómicas) y una variable objetivo de tres clases medida al final de la duración "
        "normal del programa: **Deserción**, **Matriculado** y **Graduado**. Sin valores nulos ni duplicados."
    )
    with st.expander("Diccionario de variables"):
        dic = pd.DataFrame([
            {"Variable interna": c, "Descripción": dd.FEATURE_INFO[c][0], "Unidad / codificación": dd.FEATURE_INFO[c][1],
             "Tipo": "Nominal" if c in dd.NOMINAL_FEATURES else "Binaria" if c in dd.BINARY_FEATURES else "Numérica"}
            for c in dd.ALL_FEATURES
        ])
        st.dataframe(dic, hide_index=True, width="stretch")

    st.markdown("#### 🧭 Metodología")
    cols = st.columns(len(info.METODOLOGIA))
    for i, (col, (t, d)) in enumerate(zip(cols, info.METODOLOGIA)):
        col.markdown(f"**{i + 1}. {t}**")
        col.caption(d)

    st.markdown("#### 🤖 Modelos utilizados")
    st.markdown(
        "- **MLP con *embeddings***: perceptrón multicapa (PyTorch) con activación ReLU, *dropout* y una capa de "
        "*embeddings* para las variables nominales (programa, ocupación de los padres, etc.).\n"
        "- **FT-Transformer**: cada variable se convierte en un *token* y un codificador Transformer "
        "(auto-atención multi-cabeza) los combina; la clase se lee del token [CLS].\n"
        "- **Referencia (ACA 1)**: Regresión Logística, Random Forest y HistGradientBoosting, con la misma partición "
        "y validación cruzada.\n\n"
        "Las redes se entrenan con pérdida de entropía cruzada **ponderada por clase**, optimizador AdamW y "
        "*early stopping* sobre el macro-F1 de validación, en dos modalidades: **información completa** y "
        "**alerta temprana** (sin datos del 2.º semestre)."
    )
    st.markdown("#### ⚠️ Limitaciones")
    st.markdown("\n".join(f"- {x}" for x in info.LIMITACIONES))

    st.markdown("#### 👥 Integrantes")
    if info.INTEGRANTES:
        st.markdown("\n".join(f"- {n}" for n in info.INTEGRANTES))
    else:
        st.caption("Pendiente: completar `INTEGRANTES` en `src/project_info.py`.")
    links = []
    if info.ENLACE_REPO:
        links.append(f"[Repositorio]({info.ENLACE_REPO})")
    if info.ENLACE_VIDEO:
        links.append(f"[Video]({info.ENLACE_VIDEO})")
    if links:
        st.markdown(" · ".join(links))

    st.markdown("#### 📚 Referencias")
    st.markdown("\n".join(f"- {r}" for r in info.REFERENCIAS_APA))


PAGES = {
    "Dashboard": page_dashboard,
    "Predicción individual": page_predict,
    "Comparación de modelos": page_compare,
    "Acerca del proyecto": page_about,
}
PAGES[section]()
