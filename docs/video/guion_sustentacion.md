# Guion del video de sustentación (3 a 5 minutos)

> Objetivo del enunciado: exposición sintética del equipo con **problema, arquitectura de red seleccionada, métricas obtenidas y demostración en vivo de la app en Streamlit y del póster**. Cumplir el tiempo (3–5 min) es parte de la rúbrica (15 %).
> Reparta los bloques entre los integrantes. Las cifras entre `[ ]` se toman del notebook ejecutado / `models/metricas_nn.json`.

| Tiempo | Bloque | Quién | Qué decir / mostrar |
|---|---|---|---|
| 0:00–0:30 | **Problema** | Integrante 1 | La deserción universitaria se detecta tarde. Pregunta orientadora: ¿cómo aprovechar redes neuronales para extraer patrones complejos en datos educativos y usarlos en una interfaz Streamlit? Dataset UCI 697: 4.424 estudiantes, 36 variables, 3 clases (Deserción 32 %, Matriculado 18 %, Graduado 50 %). |
| 0:30–1:15 | **Datos y preparación** | Integrante 2 | Datos en **SQLite** (`data/estudiantes.db`, consultas SQL). Misma limpieza y partición 80/20 del ACA 1. Codificación: numéricas estandarizadas + *embeddings* para las nominales (programa, ocupación de los padres). Desbalance: pérdida **ponderada por clase**. Sin fuga: todo se ajusta solo con entrenamiento. |
| 1:15–2:15 | **Arquitectura de la red** | Integrante 3 | Mostrar el diagrama del póster. **MLP con embeddings** (ReLU, dropout) y **FT-Transformer** (cada variable es un token; auto-atención; token [CLS]). Por qué no CNN/RNN/Hugging Face (datos tabulares, sin imágenes ni texto). AdamW + *early stopping* sobre macro-F1; mostrar las **curvas de pérdida** (entrenamiento vs. validación) y explicar el sobreajuste. Selección por validación cruzada de 5 pliegues. |
| 2:15–3:00 | **Métricas** | Integrante 4 | Tabla del póster: red elegida `[nombre]`, macro-F1 CV `[x.xxx]`, en prueba accuracy `[x.xxx]`, macro-F1 `[x.xxx]`. **Comparación honesta con los modelos clásicos del ACA 1: desempeño comparable, no superior** (pocos datos tabulares). Matriz de confusión: «Matriculado» es la clase más difícil. Alerta temprana (solo 1.er semestre): macro-F1 `[x.xxx]`. |
| 3:00–4:15 | **Demo en vivo** | Integrante 1 o 2 | Abrir la app en Streamlit: Dashboard → **Predicción individual** (llenar el formulario, mostrar probabilidades por clase; cambiar a *Alerta temprana*) → **Comparación de modelos** (tabla, matriz de confusión, **curvas de entrenamiento**). Mostrar brevemente el **póster PDF** (QR de app, repo y video). |
| 4:15–4:45 | **Conclusiones y ética** | Integrante 3 | Las redes igualan a los modelos clásicos; son «caja negra» (importancia por permutación solo indica asociación); datos de una institución portuguesa; sesgos posibles en variables sensibles; herramienta de apoyo, no de decisión automática. |
| 4:45–5:00 | **Cierre** | Todos | Nombres de los integrantes y docente. |

**Checklist antes de grabar:** cronometrar (< 5:00); app abierta y con los modelos cargados; póster PDF a la mano; el video debe ser **público** (probar el enlace en una ventana de incógnito, como indicó el docente) y su URL copiada en `src/project_info.py`.
