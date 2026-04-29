# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Evaluación universal de modelos
# MAGIC
# MAGIC **Autor**: Juan Carlos Alfaro Jiménez (adaptado para manufacturing quality control)
# MAGIC
# MAGIC Esta libreta actúa como un **evaluador agnóstico e independiente** y no contiene ninguna llamada a `MLflow`. Su única responsabilidad es recibir la ruta de un modelo físico ya entrenado (`PipelineModel`), aplicarlo sobre la partición de datos que se le solicite (entrenamiento, validación o prueba), calcular las métricas de rendimiento y generar las figuras de diagnóstico. Finalmente, guarda los artefactos resultantes en un volumen de `Unity Catalog`.
# MAGIC
# MAGIC ### ¿Por qué desacoplar la evaluación de la orquestadora?
# MAGIC
# MAGIC El mismo límite de memoria de `Spark Connect` que nos obligó a aislar el entrenamiento en `07_Training_Job.ipynb` aplica igualmente a la fase de inferencia: cargar múltiples modelos pesados y ejecutar transformaciones distribuidas directamente en la sesión de la libreta orquestadora consumiría el límite estricto de **1 `GB` por sesión**, provocando el fallo `ML_CACHE_SIZE_OVERFLOW_EXCEPTION`.
# MAGIC
# MAGIC La solución es idéntica: cada evaluación se delega a esta libreta secundaria mediante `dbutils.notebook.run()`. Esto levanta una **sesión de `Spark Connect` completamente limpia y nueva**, garantizando que la caché del servidor se libere por completo al terminar la ejecución, sin importar cuántos modelos estemos comparando en nuestro flujo `MLOps`.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Importaciones y carga de utilidades compartidas
# MAGIC
# MAGIC El *script* `07_Utils.py` actúa como nuestra caja de herramientas central. A diferencia de la libreta de entrenamiento, en esta sesión **sí haremos un uso intensivo de toda la lógica matemática de evaluación y de las funciones de generación de figuras de diagnóstico** (curvas *PR*, curvas *ROC*, matrices de confusión y curvas de calibración) que dejamos allí encapsuladas.
# MAGIC
# MAGIC Este script ha sido adaptado específicamente para el contexto de **detección de defectos en componentes electrónicos**, manteniendo la misma arquitectura que el baseline de fraude.
# MAGIC
# MAGIC A continuación, importamos las librerías necesarias **exclusivamente** para la carga del modelo, la inferencia y la serialización de resultados.

# COMMAND ----------

exec(open("07_Utils.py").read(), globals())

# COMMAND ----------

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Recepción de parámetros
# MAGIC
# MAGIC Al igual que en `07_Training_Job.ipynb`, la libreta orquestadora inyecta todos los valores mediante el parámetro `arguments` de `dbutils.notebook.run()`, y todos llegan como cadenas de texto que deben convertirse explícitamente:
# MAGIC
# MAGIC * **`model_artifact_uri`**: `URI` del artefacto del modelo.
# MAGIC * **`evaluation_dataset`**: Partición sobre la que se evaluará el modelo: `train`, `validation` o `test`. En nuestro contexto, evaluamos la capacidad del modelo para detectar defectos.
# MAGIC * **`evaluation_tag`**: Identificador único de esta evaluación, usado para nombrar los directorios de artefactos en el volumen.

# COMMAND ----------

dbutils.widgets.text("model_artifact_uri", "")
dbutils.widgets.text("evaluation_dataset", "train")
dbutils.widgets.text("evaluation_tag", "train")

model_artifact_uri = dbutils.widgets.get("model_artifact_uri")
evaluation_dataset = dbutils.widgets.get("evaluation_dataset")
evaluation_tag     = dbutils.widgets.get("evaluation_tag")

print(f"Model artifact URI: {model_artifact_uri}")
print(f"Evaluation dataset: {evaluation_dataset}")
print(f"Evaluation tag: {evaluation_tag}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Obtención del conjunto de evaluación
# MAGIC
# MAGIC Gracias a la importación inicial de nuestro *script* de utilidades, las tres particiones temporales del proyecto (`train_df`, `validation_df`, `test_df`) ya se encuentran pre-cargadas en el entorno global de la libreta.
# MAGIC
# MAGIC A diferencia del proceso de entrenamiento (que fusiona particiones para maximizar el aprendizaje), esta libreta actúa como un juez estricto. Utilizando el parámetro `evaluation_dataset`, seleccionará dinámicamente la partición correspondiente a la fase del ciclo de vida en curso:
# MAGIC
# MAGIC * **`train`**: Evalúa sobre el conjunto de entrenamiento. Se utiliza para calcular la brecha de generalización junto con la evaluación sobre validación.
# MAGIC * **`validation`**: Evalúa sobre el conjunto de validación. Es el modo estándar durante el *grid search*.
# MAGIC * **`test`**: Evalúa sobre el conjunto de prueba completamente reservado. Se reserva para la comparación final entre `challenger` y `champion`.

# COMMAND ----------

if evaluation_dataset == "train":
    dataset_description = "Evaluating on the strict temporal training partition."
    eval_df = train_df
elif evaluation_dataset == "validation":
    dataset_description = "Evaluating on the validation partition."
    eval_df = validation_df
else:
    dataset_description = "Evaluating on the held-out test partition."
    eval_df = test_df

print(dataset_description)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Carga del modelo y generación de predicciones
# MAGIC
# MAGIC Se carga el `ScikitPipelineWrapper` guardado con `joblib` desde el volumen de `Unity Catalog`.
# MAGIC A continuación, se aplica sobre la partición seleccionada para obtener probabilidades y predicciones.

# COMMAND ----------

# --- Resolver la ruta local del modelo ---
# El artefacto puede llegar como "dbfs:/Volumes/..." o "/Volumes/..."
model_file_path = model_artifact_uri.replace("dbfs:", "/dbfs")

# Añadir el nombre del fichero si solo se recibe el directorio
if model_file_path.endswith("/"):
    model_file_path = model_file_path + "sklearn_model.pkl"
elif not model_file_path.endswith(".pkl"):
    model_file_path = model_file_path + "/sklearn_model.pkl"

pipeline_model = joblib.load(model_file_path)
print(f"Model loaded from: {model_file_path}")

# --- Convertir la partición a pandas y extraer features/labels ---
# Solo se necesitan las columnas de features y label; el resto no es relevante aquí.
eval_pandas_df = (
    eval_df
    .filter(F.col(label_column).isNotNull())   # Excluir inspecciones sin etiqueta (delayed feedback)
    .select([features_column, label_column])
    .toPandas()
)

# Desempaquetar el vector de features (columna de listas) a matriz numpy
X_eval = np.array(
    [np.array(x, dtype=float) if hasattr(x, "__iter__") else [float(x)]
     for x in eval_pandas_df[features_column].values]
)
y_eval = eval_pandas_df[label_column].values.astype(int)

# --- Inferencia ---
y_pred_proba = pipeline_model.predict_proba(X_eval)   # shape: (n, 2)
y_pred       = pipeline_model.predict(X_eval)
p_eval       = y_pred_proba[:, 1]                     # probabilidad de defecto (clase 1)

print(f"Evaluation set size: {len(y_eval):,} rows")
print(f"Defective ratio: {y_eval.mean():.3%}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Cálculo de métricas
# MAGIC
# MAGIC Se calculan las métricas de clasificación directamente con `scikit-learn` sobre arrays numpy,
# MAGIC evitando así cualquier dependencia con los evaluadores de `MLlib` (incompatibles con predicciones pandas).

# COMMAND ----------

# FIX: compute_metrics en 07_Utils.py usa BinaryClassificationEvaluator de MLlib,
# que requiere un Spark DataFrame con columna "rawPrediction". Como aquí trabajamos
# con sklearn/numpy, recalculamos las métricas directamente.

eval_metrics = {
    "auc_pr":    float(average_precision_score(y_eval, p_eval)),
    "auc_roc":   float(roc_auc_score(y_eval, p_eval)),
    "f1":        float(f1_score(y_eval, y_pred, zero_division=0)),
    "recall":    float(recall_score(y_eval, y_pred, zero_division=0)),
    "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
    "accuracy":  float(accuracy_score(y_eval, y_pred)),
}

print(f"AUC-PR:    {eval_metrics['auc_pr']:.4f}")
print(f"AUC-ROC:   {eval_metrics['auc_roc']:.4f}")
print(f"F1-score:  {eval_metrics['f1']:.4f}")
print(f"Recall:    {eval_metrics['recall']:.4f}")
print(f"Precision: {eval_metrics['precision']:.4f}")
print(f"Accuracy:  {eval_metrics['accuracy']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Umbral de decisión óptimo
# MAGIC
# MAGIC Se calcula el umbral de decisión óptimo mediante `find_best_threshold`.
# MAGIC Este umbral es el valor de corte sobre la probabilidad que maximiza el F1-score.

# COMMAND ----------

best_threshold, best_f1_score = find_best_threshold(y_eval, p_eval)

print(f"Best threshold: {best_threshold:.2f}")
print(f"Best F1-score:  {best_f1_score:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 7. Generación y guardado de figuras
# MAGIC
# MAGIC Se generan las figuras de diagnóstico sobre la partición evaluada y se guardan como archivos `.png` en el volumen temporal de `Unity Catalog`.

# COMMAND ----------

eval_tmp_path   = str(Path(uc_volume_path) / "evaluations" / evaluation_tag)
figures_local_path = str(Path(eval_tmp_path) / "figures")
dbutils.fs.mkdirs(figures_local_path)

save_diagnostic_figure(
    fig_pr_curve(y_eval, p_eval, eval_metrics["auc_pr"],
                 f"PR — {evaluation_tag} ({evaluation_dataset})"),
    figures_local_path, "pr_curve.png"
)
save_diagnostic_figure(
    fig_roc_curve(y_eval, p_eval, eval_metrics["auc_roc"],
                  f"ROC — {evaluation_tag} ({evaluation_dataset})"),
    figures_local_path, "roc_curve.png"
)
save_diagnostic_figure(
    fig_confusion_matrix(y_eval, y_pred,
                         f"Confusion matrix — {evaluation_tag} ({evaluation_dataset})"),
    figures_local_path, "confusion_matrix.png"
)
save_diagnostic_figure(
    fig_calibration_curve(y_eval, p_eval,
                          f"Calibration — {evaluation_tag} ({evaluation_dataset})"),
    figures_local_path, "calibration_curve.png"
)
save_diagnostic_figure(
    fig_threshold_sweep(y_eval, p_eval,
                        f"Threshold sweep — {evaluation_tag} ({evaluation_dataset})"),
    figures_local_path, "threshold_sweep.png"
)

print("All diagnostic figures generated and saved successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 8. Informe de clasificación
# MAGIC
# MAGIC La función `classification_report` genera una tabla detallada con *precision*, *recall* y *F1-score* desglosados por clase. El informe se guarda como fichero de texto plano (`.txt`) en el volumen de `Unity Catalog`.

# COMMAND ----------

target_names = ["Good", "Defective"]

report_path = str(Path(eval_tmp_path) / "classification_report.txt")

with open(report_path, "w") as fh:
    fh.write(classification_report(y_eval, y_pred, target_names=target_names))

print(f"Classification report successfully saved to {report_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 9. Liberación de memoria y retorno del resultado
# MAGIC
# MAGIC Mismo patrón que en `07_Training_Job.ipynb`, devolviendo en este caso las métricas de evaluación
# MAGIC y las rutas de los artefactos generados sobre la partición seleccionada.

# COMMAND ----------

# FIX: las variables originales (eval_predictions, lr_fitted) no existen en este notebook.
# Solo se eliminan los objetos que sí se crearon aquí.
del pipeline_model, X_eval, y_pred_proba, y_pred, p_eval, eval_pandas_df
gc.collect()

result = {
    "evaluation_tag":      evaluation_tag,
    "evaluation_dataset":  evaluation_dataset,
    "eval_metrics":        eval_metrics,
    "figures_local_path":  figures_local_path,
    "report_path":         report_path,
    "best_threshold":      float(best_threshold),
    "best_f1_at_threshold": float(best_f1_score),
}

print(f"Exiting notebook and returning results for evaluation: {evaluation_tag}")
dbutils.notebook.exit(json.dumps(result))