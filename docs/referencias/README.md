# Referencias del proyecto

- `ACA_1_Aprendizaje_Automatico_CUN.pdf` — enunciado oficial del Proyecto ACA N.º 1 (CUN, Aprendizaje
  Automático, Dr. Hamilton Rivera Flor). Pilar elegido: **Proyecto 2 — Educación**.

## Correspondencia con el enunciado

| Requisito del ACA | Dónde se cumple |
|---|---|
| Paso 1 · Exploración y EDA | `notebooks/ACA1_Student_Dropout.ipynb`, Parte 1 |
| Paso 2 · Limpieza y datos (preprocesamiento, imputación) | Notebook, Parte 2 · `src/data_loader.py` |
| Paso 3 · Modelado clásico | Notebook, Partes 3–4 · `src/train.py`, `src/preprocessing.py` |
| Paso 4 · Evaluación y métricas | Notebook, Parte 4 · `src/evaluate.py` · `models/metricas*.json` |
| Paso 5 · Despliegue Streamlit | `app.py`, `requirements.txt` |
| Entregable 1 · Notebook reproducible, modular y comentado | `notebooks/ACA1_Student_Dropout.ipynb` (ejecutado) |
| Entregable 2 · Póster infografía (APA 7) | `docs/poster/poster_EduPredictAI.pdf` |
| Entregable 3 · App Streamlit + requirements | `app.py`, `src/`, `requirements.txt` |
| Entregable 4 · Video de 3 a 5 min | https://www.youtube.com/watch?v=BEmN8dqMv-o · guion en `docs/video/guion_sustentacion.md` |

## Nota sobre la cita del dataset

El enunciado cita el dataset como *Martins, M. V., Real, M. B., Machado, J., & Becker, A. (2021)*. La cita
oficial que publica el UCI Machine Learning Repository es:

Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic
Success* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

El proyecto usa la cita oficial de UCI. Si el docente exige la forma del enunciado, conviene confirmarlo con él.
