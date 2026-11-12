"""Textos académicos del proyecto (sección "Acerca del proyecto" y póster).

Completar ``INTEGRANTES``, ``DOCENTE`` y los enlaces públicos cuando estén
disponibles. No se rellenan con datos inventados.
"""

NOMBRE = "EduPredict AI"
DESCRIPCION = (
    "Sistema inteligente de análisis académico y predicción de deserción universitaria "
    "mediante redes neuronales (MLP y FT-Transformer) desplegadas en Streamlit."
)

CONTEXTO = (
    "Proyecto ACA N.º 2 de la asignatura Aprendizaje Automático — Modelado con redes neuronales "
    "artificiales (CNN, RNN, Transformers) y despliegue interactivo en Streamlit. Especialización en "
    "Inteligencia Artificial, Escuela de Ingeniería, Corporación Unificada Nacional de "
    "Educación Superior (CUN)."
)

PILAR = "Proyecto 2 — Educación: analítica académica y prevención de la deserción"

PREGUNTA = (
    "¿Cómo transformar datos educativos en conocimiento predictivo para anticipar la "
    "deserción de un estudiante y facilitar la identificación de posibles situaciones "
    "de riesgo académico?"
)

OBJETIVO = (
    "Desarrollar e implementar redes neuronales (un perceptrón multicapa con embeddings y un "
    "Transformer para datos tabulares) que clasifiquen la situación final de un estudiante "
    "(deserción, matriculado o graduado), evaluar su desempeño frente a los modelos clásicos del "
    "ACA 1 y desplegar el mejor modelo en una aplicación interactiva de Streamlit."
)

# Completar con los nombres completos de los integrantes, p. ej. ["Nombre Apellido", ...]
INTEGRANTES: list[str] = [
    "Paula Carolina Calvo Pérez",
    "Juan Sebastián Caviedes Roa",
    "Jonathan Camilo Flórez Bolívar",
    "Daniel Santiago Quintero Niño",
]
DOCENTE: str = "Dr. Hamilton Rivera Flor"

# Completar cuando estén publicados (la app y el póster muestran un espacio reservado si están vacíos).
ENLACE_APP: str = ""     # URL pública en Streamlit Community Cloud
ENLACE_VIDEO: str = ""   # URL pública del video de sustentación (3 a 5 min)
ENLACE_REPO: str = ""    # URL del repositorio de GitHub del ACA 2

# Ruta de aprendizaje del ACA (enunciado oficial) y su implementación en este proyecto
RUTA_ACA = [
    ("Paso 1 · Exploración y gestión SQL", "Notebook, Parte 1 · src/db.py · data/estudiantes.db"),
    ("Paso 2 · Preparación y pipeline", "Notebook, Parte 2 · src/nn/data.py"),
    ("Paso 3 · Modelado neuronal", "Notebook, Partes 3–5 · src/nn/models.py · src/nn/training.py"),
    ("Paso 4 · Evaluación y métricas", "Notebook, Partes 6–7 · src/evaluate.py · models/metricas_nn*.json"),
    ("Paso 5 · Despliegue Streamlit", "app.py · src/predict.py"),
]

METODOLOGIA = [
    ("Datos en SQL", "El CSV de UCI 697 se carga en SQLite (data/estudiantes.db); el EDA y la app consultan la base con SQL."),
    ("Limpieza determinista", "Normalización de nombres, tipos y verificación de nulos, duplicados, rangos y reglas lógicas (ACA 1)."),
    ("Partición", "División estratificada 80/20 (semilla 42), la misma del ACA 1. El 20 % de prueba queda aislado hasta la evaluación final."),
    ("Codificación para redes", "Numéricas y binarias estandarizadas (+ tasas de aprobación); nominales como índices para capas de embeddings. Todo se ajusta solo con entrenamiento."),
    ("Entrenamiento", "Pérdida de entropía cruzada ponderada por clase, AdamW, dropout y early stopping sobre el macro-F1 de validación."),
    ("Selección", "Validación cruzada estratificada de 5 pliegues (macro-F1) entre configuraciones de MLP y FT-Transformer."),
    ("Evaluación y despliegue", "Loss, accuracy, F1, matriz de confusión y curvas en prueba; la red ganadora se guarda en .pt y se sirve en Streamlit."),
]

LIMITACIONES = [
    "Con ~4.400 filas tabulares, las redes neuronales no superan de forma clara a los modelos de árboles del ACA 1; su ventaja es la flexibilidad de la arquitectura, no un salto de desempeño.",
    "Las redes neuronales son modelos de «caja negra»: la importancia por permutación aproxima qué variables pesan, pero no explica cómo decide la red en cada caso.",
    "Los datos provienen de una sola institución de educación superior de Portugal (cohortes 2008/09–2018/19); los patrones pueden no trasladarse a otros contextos, como el colombiano.",
    "La clase «Matriculado» (estudiantes que siguen inscritos al final de la duración normal del programa) es la minoritaria y la más difícil de distinguir.",
    "Las probabilidades del modelo no están calibradas para uso institucional ni constituyen una medición validada de riesgo; son estimaciones estadísticas con fines académicos.",
    "El modelo identifica asociaciones, no causas. La importancia de una variable no implica que modificarla cambie el resultado del estudiante.",
    "Las variables sensibles (sexo, nacionalidad, estado civil) pueden introducir sesgos; cualquier uso real requeriría un análisis de equidad.",
    "Los indicadores macroeconómicos corresponden al año de ingreso y solo toman 10 combinaciones distintas en el dataset.",
]

REFERENCIAS_APA = [
    # APA 7.ª edición, orden alfabético. Los *asteriscos* marcan cursiva (Markdown en la app, <i> en el póster).
    "Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*, 3146–3154.",
    "Martins, M. V., Tolledo, D., Machado, J., Baptista, L. M. T., & Realinho, V. (2021). Early prediction of student's performance in higher education: A case study. En Á. Rocha, H. Adeli, G. Dzemyda, F. Moreira, & A. M. Ramalho Correia (Eds.), *Trends and applications in information systems and technologies* (pp. 166–175). Springer. https://doi.org/10.1007/978-3-030-72657-7_16",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html",
    "Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). Predicting student dropout and academic success. *Data, 7*(11), Artículo 146. https://doi.org/10.3390/data7110146",
    "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict students' dropout and academic success* [Conjunto de datos]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89",
    "Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. *Advances in Neural Information Processing Systems, 34*, 18932–18943.",
    "Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1711.05101",
    "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems, 32*, 8024–8035.",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*, 5998–6008.",
    "Streamlit Inc. (2026). *Streamlit* (Versión 1.66.1) [Software]. https://streamlit.io",
]

# APA 7: orden alfabético por primer autor
REFERENCIAS_APA.sort(key=lambda r: r.lower())
