"""Redes neuronales (PyTorch) para EduPredict AI — ACA N.º 2.

* ``data``     : codificador tabular (numéricas escaladas + índices categóricos).
* ``models``   : MLP con embeddings y FT-Transformer (Transformer para tablas).
* ``training`` : bucle de entrenamiento con pesos de clase y *early stopping*,
                 validación cruzada y curvas de aprendizaje.
* ``wrapper``  : ``NNClassifier`` (interfaz tipo scikit-learn) y serialización.
* ``explain``  : importancia por permutación para cualquier clasificador.

El módulo ``src.nn.data`` no depende de PyTorch; el resto sí.
"""
