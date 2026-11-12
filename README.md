# 🎓 EduPredict AI — ACA N.º 2 (Redes neuronales)

**Predicción de deserción universitaria con redes neuronales (MLP con *embeddings* y FT-Transformer) desplegadas en Streamlit.**

Proyecto ACA N.º 2 — Aprendizaje Automático · Especialización en Inteligencia Artificial · Corporación Unificada Nacional de Educación Superior (CUN). Continuación del [ACA 1](https://github.com/DanielSQN/edupredict-ml-streamlit): mismo dataset, misma limpieza, misma partición 80/20 y misma validación cruzada, de modo que las redes se comparan de forma justa con los modelos clásicos.

## 1. Qué hay en este repositorio

| Entregable del ACA 2 | Dónde está |
|---|---|
| 📓 Notebook reproducible | [`notebooks/ACA2_Student_Dropout_NN.ipynb`](notebooks/ACA2_Student_Dropout_NN.ipynb) |
| 💻 App Streamlit | [`app.py`](app.py) · `src/` · [`requirements.txt`](requirements.txt) |
| 📄 Póster infografía (PDF, APA 7) | se genera con `python scripts/build_poster.py` → `docs/poster/` |
| 🎥 Video de sustentación (3–5 min) | guion en [`docs/video/guion_sustentacion.md`](docs/video/guion_sustentacion.md) |
| 📋 Enunciado y rúbrica | [`docs/referencias/ACA_2_Aprendizaje_Automatico_CUN.pdf`](docs/referencias/ACA_2_Aprendizaje_Automatico_CUN.pdf) |

## 2. Problema, objetivo y dataset

**Pregunta orientadora.** ¿Cómo aprovechar arquitecturas de redes neuronales para extraer patrones complejos en datos educativos, optimizando el rendimiento predictivo y permitiendo su interacción práctica a través de una interfaz en Streamlit?

**Objetivo general.** Desarrollar e implementar redes neuronales (un perceptrón multicapa con *embeddings* y un Transformer para datos tabulares) que clasifiquen la situación final de un estudiante (deserción, matriculado o graduado), evaluar su desempeño frente a los modelos clásicos del ACA 1 y desplegar la mejor en una aplicación Streamlit.

**Dataset.** [*Predict Students' Dropout and Academic Success*](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI 697; Realinho et al., 2021): 4.424 estudiantes, 36 variables (9 nominales, 8 binarias, 19 numéricas), 3 clases (Deserción 32,1 %, Matriculado 17,9 %, Graduado 49,9 %), sin nulos ni duplicados. Diccionario completo en `src/data_dictionary.py`.

## 3. Metodología (los 5 pasos del enunciado)

1. **Exploración y gestión SQL** — el CSV se carga en **SQLite** (`data/estudiantes.db`, módulo `src/db.py`). SQLite viene con Python: no requiere cuenta, servidor ni pago. El EDA y la app consultan la base con SQL.
2. **Preparación y pipeline** (`src/nn/data.py`) — numéricas y binarias estandarizadas + tasas de aprobación por semestre; variables nominales como índices para capas de *embeddings*. Todo se ajusta solo con entrenamiento (sin fuga).
3. **Modelado neuronal** (`src/nn/models.py`, `src/nn/training.py`) — dos arquitecturas en PyTorch:
   - **MLP con *embeddings***: capas densas + ReLU + dropout.
   - **FT-Transformer** (Gorishniy et al., 2021): cada variable es un *token*; codificador Transformer con auto-atención; salida desde el token [CLS].
   - Pérdida de entropía cruzada **ponderada por clase** (desbalance), optimizador **AdamW**, *early stopping* sobre el macro-F1 de validación.
4. **Evaluación** — validación cruzada estratificada de 5 pliegues (macro-F1) para elegir configuración y arquitectura; en prueba: loss, accuracy, balanced accuracy, macro-F1, precisión/recall/F1 por clase, ROC-AUC, matriz de confusión, **curvas de entrenamiento/validación** e importancia por permutación. Comparación con Regresión Logística, Random Forest y HistGradientBoosting del ACA 1.
5. **Despliegue** — `app.py` (Streamlit) carga `models/modelo_nn*.pt` y ofrece Dashboard, Predicción individual (dos escenarios: *completo* y *alerta temprana*), Comparación de modelos (con curvas) y Acerca del proyecto.

> **¿Por qué no CNN, RNN o Hugging Face?** Los datos son una tabla (sin imágenes, señales ni texto). Las CNN/RNN y los modelos de Hugging Face (BERT, ViT…) están diseñados para esas modalidades. Para tablas, las arquitecturas apropiadas son el MLP y el FT-Transformer (la versión tabular de la arquitectura Transformer).

## 4. Cómo reproducir

### Opción A — Google Colab (recomendada si no tiene PyTorch local)
1. Suba la carpeta del proyecto a GitHub (o a su Drive) y ábrala en Colab; si usa GitHub, pegue la URL del repo en `REPO_URL` (celda 0 del notebook).
2. *Entorno de ejecución → CPU* es suficiente. Ejecute **Run all**.
3. Al terminar, descargue la carpeta `models/` (contiene `modelo_nn*.pt`, `metadata_nn*.json`, `metricas_nn*.json`) y cópiela al repositorio.

### Opción B — Local
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
jupyter notebook notebooks/ACA2_Student_Dropout_NN.ipynb   # Run all (genera models/)
streamlit run app.py
pytest                                                      # pruebas (se omiten las que necesitan models/ si aún no existen)
```
El notebook tiene dos interruptores en la celda de configuración:
- `QUICK = True/False`: `True` hace una prueba rápida (~5 min); `False` es la ejecución completa (~15–30 min en CPU).
- `LIMPIEZA_COMPLETA = True/False`: `True` muestra paso a paso la limpieza del ACA 1 (y verifica que coincide con `clean_dataset`); `False` solo muestra un resumen/auditoría.

### Despliegue en Streamlit Community Cloud
1. Suba el repositorio a GitHub **incluyendo `models/*.pt`, `models/*.json` y `data/estudiantes.db`**.
2. En [share.streamlit.io](https://share.streamlit.io) cree la app apuntando a `app.py`.
3. `requirements.txt` instala la versión **CPU** de PyTorch (línea `--extra-index-url`). Si el despliegue falla por esa línea, elimínela y deje solo `torch`.
4. Copie la URL pública en `src/project_info.py` (`ENLACE_APP`, `ENLACE_REPO`, `ENLACE_VIDEO`).

## 5. Estructura

```
aca_2_machine_learning/
├── app.py                         # App Streamlit (no entrena; carga models/*.pt)
├── requirements.txt / requirements-dev.txt
├── data/
│   ├── data.csv                   # CSV original UCI 697
│   ├── estudiantes_limpio.csv     # Limpieza del ACA 1
│   └── estudiantes.db             # SQLite (tablas estudiantes_raw, estudiantes; vista resumen_target)
├── notebooks/
│   ├── ACA2_Student_Dropout_NN.ipynb   # ← notebook del ACA 2
│   └── aca1/ACA1_Student_Dropout.ipynb # referencia (ACA 1)
├── src/
│   ├── db.py                      # SQLite: creación y consultas
│   ├── data_dictionary.py, data_loader.py, preprocessing.py, train.py, evaluate.py   # base del ACA 1 (limpieza, partición, métricas)
│   ├── predict.py                 # carga del .pt y predicción (usado por la app y las pruebas)
│   ├── project_info.py            # textos, integrantes, enlaces, referencias APA
│   └── nn/
│       ├── data.py                # TabularEncoder (z-score + índices para embeddings)
│       ├── models.py              # MLPNet y FTTransformer
│       ├── training.py            # entrenamiento, early stopping, CV, exportación
│       ├── wrapper.py             # NNClassifier (predict_proba / save / load)
│       └── explain.py             # importancia por permutación
├── models/
│   ├── modelo_nn.pt, metadata_nn.json, metricas_nn.json                      # escenario completo   (se generan al ejecutar el notebook)
│   ├── modelo_nn_temprano.pt, metadata_nn_temprano.json, metricas_nn_temprano.json   # alerta temprana
│   └── aca1/                      # métricas de los modelos clásicos del ACA 1 (referencia de comparación)
├── scripts/build_poster.py        # póster HTML → PDF desde las métricas reales
├── tests/                         # pytest: codificador, redes, predicción, arranque de Streamlit
└── docs/                          # enunciado, guion del video, capturas, póster
```

## 6. Resultados

> ⚠️ **Estado:** el notebook está entregado como borrador **sin ejecutar** (los modelos `models/*.pt` aún no existen). Se completan al ejecutar el notebook: ver la tabla de la Parte 7 del notebook y la página **Comparación de modelos** de la app (todas las cifras se leen de `models/metricas_nn*.json`).

## 7. Limitaciones y ética

- ~4.400 filas tabulares de una sola institución portuguesa: las redes **no superan claramente** a los modelos de árboles del ACA 1.
- Las redes son «cajas negras»; la importancia por permutación indica asociación, no causalidad ni explicación individual.
- Las probabilidades no están calibradas ni validadas como medición de riesgo; variables sensibles (sexo, nacionalidad, estado civil) pueden introducir sesgos. Herramienta de apoyo académico, no de decisión automática.

## 8. Referencias (APA 7)
Ver `src/project_info.py → REFERENCIAS_APA` (también se muestran en la app y el póster).
