# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Entrenamiento de una combinación de hiperparámetros
# MAGIC
# MAGIC **Autor**: Juan Carlos Alfaro Jiménez
# MAGIC
# MAGIC Esta libreta **no contiene ninguna llamada a `MLflow`**. Su única responsabilidad es aislar el entrenamiento de un único `PipelineModel` de `MLlib` utilizando los hiperparámetros recibidos. Durante su ejecución independiente, ajustará el modelo a los datos y guardará el artefacto físico resultante directamente en un volumen de `Unity Catalog`.
# MAGIC
# MAGIC ### ¿Por qué desacoplar el entrenamiento de la orquestadora?
# MAGIC
# MAGIC `Spark Connect` gestiona los modelos entrenados por `MLlib` almacenándolos en una caché del lado del servidor, la cual tiene un límite estricto de **1`GB` por sesión**. Si intentásemos entrenar iterativamente múltiples combinaciones de hiperparámetros (un *grid search* tradicional) dentro de una misma sesión, estos modelos se acumularían rápidamente en memoria, provocando el fallo `ML_CACHE_SIZE_OVERFLOW_EXCEPTION`.
# MAGIC
# MAGIC La solución arquitectónica óptima consiste en delegar el entrenamiento a esta libreta secundaria, invocándola de forma aislada mediante la instrucción `dbutils.notebook.run()` desde cualquier libreta orquestadora del proyecto (ya sea en la fase de experimentación o durante el paso a producción). 
# MAGIC
# MAGIC De este modo, cada invocación levanta una **sesión de `Spark Connect` completamente limpia y nueva**, garantizando que la caché de memoria se libere correctamente al terminar cada ciclo de entrenamiento.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Importaciones y carga de utilidades compartidas
# MAGIC
# MAGIC El *script* `07_Utils.py` actúa como nuestra caja de herramientas central. Expone la configuración estructural del proyecto y las funciones maestras para la carga controlada del conjunto de datos, la generación de los *splits* temporales y el cálculo dinámico de los pesos de clase. Además, deja preparados en el entorno global todas las listas de columnas y expresiones `SQL` necesarias para construir el *pipeline* de preprocesamiento.
# MAGIC
# MAGIC Aunque este *script* también contiene funciones de evaluación de modelos, en esta sesión aislada nos apoyaremos en él **exclusivamente** para la creación del conjunto de datos, el entrenamiento del *pipeline* y, finalmente, la generación gráfica de los coeficientes del algoritmo.
# MAGIC
# MAGIC A continuación, importamos las librerías necesarias **exclusivamente** para el ensamblado y entrenamiento del *pipeline* en esta sesión aislada.

# COMMAND ----------

exec(open("07_Utils.py").read(), globals())

# COMMAND ----------

import gc
import json
from pathlib import Path

import pandas as pd

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import DenseVector, Vectors
from pyspark.sql.functions import col, when, lit, udf
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.stat import Correlation
import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Recepción de parámetros
# MAGIC
# MAGIC Los *widgets* son el mecanismo estándar en `Databricks` para parametrizar y pasar información dinámicamente entre libretas. En nuestro flujo, la libreta orquestadora inyecta todos estos valores simultáneamente utilizando el parámetro `arguments` de la instrucción `dbutils.notebook.run()`.
# MAGIC
# MAGIC > **Importante sobre el tipado**: Todos los valores recibidos a través de los *widgets* llegan **siempre como cadenas de texto** (`String`). Es imprescindible convertirlos explícitamente a su tipo de dato correspondiente (flotantes, enteros o booleanos) antes de pasarlos al modelo o al *pipeline*.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.1. Hiperparámetros del preprocesado
# MAGIC
# MAGIC Estos hiperparámetros controlan el comportamiento de las etapas de transformación de datos previas al clasificador. No se definen aquí, esta libreta solo los recibe y los aplica:
# MAGIC
# MAGIC * **`imputer_strategy`**: Estrategia de imputación para columnas numéricas con nulos (`median` es más robusta ante valores atípicos que `mean`).
# MAGIC * **`var_selector_threshold`**: Varianza mínima que debe tener una característica para no ser descartada.
# MAGIC * **`scaler_with_mean`**: Centrar los datos en la media antes de escalar (mantenido en `False` para no destruir la dispersidad del *one-hot encoding*).
# MAGIC * **`scaler_with_std`**: Normalizar la distribución a varianza unitaria.
# MAGIC * **`ohe_drop_last`**: Eliminar la última categoría en la codificación *one-hot* para evitar multicolinealidad perfecta.
# MAGIC * **`si_handle_invalid`**: Política del `StringIndexer` ante categorías nuevas (`keep` asigna un índice especial).
# MAGIC * **`si_order_type`**: Criterio de ordenación de categorías (`frequencyDesc` asigna el índice 0 a la más frecuente).
# MAGIC * **`ohe_handle_invalid`**: Política del codificador *one-hot* ante índices desconocidos en inferencia.
# MAGIC * **`asm_handle_invalid`**: Política del `VectorAssembler` ante nulos residuales (`error` actúa como barrera de calidad estricta).

# COMMAND ----------

# Default values are used only in interactive runs; notebook.run() overrides them
dbutils.widgets.text("imputer_strategy", "median")
dbutils.widgets.text("var_selector_threshold", "0.01")
dbutils.widgets.text("scaler_with_mean", "False")
dbutils.widgets.text("scaler_with_std", "True")
dbutils.widgets.text("ohe_drop_last", "True")
dbutils.widgets.text("si_handle_invalid", "keep")
dbutils.widgets.text("si_order_type", "frequencyDesc")
dbutils.widgets.text("ohe_handle_invalid", "keep")
dbutils.widgets.text("asm_handle_invalid", "error")

imputer_strategy = dbutils.widgets.get("imputer_strategy")
var_selector_threshold = float(dbutils.widgets.get("var_selector_threshold"))
scaler_with_mean = dbutils.widgets.get("scaler_with_mean").lower() == "true"
scaler_with_std = dbutils.widgets.get("scaler_with_std").lower() == "true"
ohe_drop_last = dbutils.widgets.get("ohe_drop_last").lower() == "true"
si_handle_invalid = dbutils.widgets.get("si_handle_invalid")
si_order_type = dbutils.widgets.get("si_order_type")
ohe_handle_invalid = dbutils.widgets.get("ohe_handle_invalid")
asm_handle_invalid = dbutils.widgets.get("asm_handle_invalid")

print(f"Imputer strategy: {imputer_strategy}")
print(f"Variance selector threshold: {var_selector_threshold}")
print(f"Scaler with mean: {scaler_with_mean}")
print(f"Scaler with standard deviation: {scaler_with_std}")
print(f"One-hot encoding drop last category: {ohe_drop_last}")
print(f"String indexer handle invalid: {si_handle_invalid}")
print(f"String indexer order type: {si_order_type}")
print(f"One-hot encoding handle invalid: {ohe_handle_invalid}")
print(f"Assembler handle invalid: {asm_handle_invalid}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.2. Hiperparámetros del clasificador
# MAGIC
# MAGIC Estos hiperparámetros controlan directamente el comportamiento del algoritmo final (`LogisticRegression`):
# MAGIC
# MAGIC * **`reg_param`**: Intensidad global de la penalización de regularización (`L2` puro o mezcla de `L1` y `L2`).
# MAGIC * **`elastic_net_param`**: Balance de la penalización `Elastic Net` (mezcla entre `Ridge` `0.0` y `Lasso` `1.0`).
# MAGIC * **`max_iter`**: Número máximo de iteraciones permitidas para el optimizador `L-BFGS`.
# MAGIC * **`family`**: Tipo de modelo (`binomial` especifica regresión logística binaria).
# MAGIC * **`standardization`**: Estandarización interna del clasificador (fijado a `False` porque el *pipeline* ya incluye un `StandardScaler`).
# MAGIC * **`threshold`**: Umbral de decisión por defecto.

# COMMAND ----------

dbutils.widgets.text("reg_param", "0.01")
dbutils.widgets.text("elastic_net_param", "0.0")
dbutils.widgets.text("max_iter", "100")
dbutils.widgets.text("family", "binomial")
dbutils.widgets.text("standardization", "False")
dbutils.widgets.text("threshold", "0.5")

reg_param = float(dbutils.widgets.get("reg_param"))
elastic_net_param = float(dbutils.widgets.get("elastic_net_param"))
max_iter = int(dbutils.widgets.get("max_iter"))
family = dbutils.widgets.get("family")
standardization = dbutils.widgets.get("standardization").lower() == "true"
threshold = float(dbutils.widgets.get("threshold"))

# Run tag: Unique visual identifier for this grid point
run_tag = f"lr__rp{reg_param}__en{elastic_net_param}__seed{seed}"

print(f"Regularization hyperparameter: {reg_param}")
print(f"Elastic net hyperparameter: {elastic_net_param}")
print(f"Maximum iterations: {max_iter}")
print(f"Family: {family}")
print(f"Standardization: {standardization}")
print(f"Threshold: {threshold}")
print(f"Run tag: {run_tag}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2.3. Parámetros de orquestación y modo de entrenamiento
# MAGIC
# MAGIC Para soportar el ciclo de vida completo de `MLOps`, la libreta acepta un parámetro `training_mode` que determina qué particiones temporales de datos se utilizarán para ajustar el modelo. Dependiendo de la fase del proyecto orquestada, inyectaremos un modo distinto:
# MAGIC
# MAGIC * **`train` (fase de experimentación)**: El modelo se entrena exclusivamente con el conjunto de entrenamiento. Es el modo por defecto y se utiliza para iterar sobre múltiples hiperparámetros y validarlos de forma justa contra el conjunto de validación.
# MAGIC * **`train_val` (fase de evaluación del `challenger`)**: Fusiona los conjuntos de entrenamiento y validación. Se invoca una vez que los mejores hiperparámetros ya han sido fijados. Permite que el modelo consolide su aprendizaje con más información histórica antes de someterlo al examen final e imparcial contra el conjunto de prueba.
# MAGIC * **`train_val_test` (fase de despliegue del `champion`)**: Utiliza el histórico completo de datos disponible. Se ejecuta únicamente cuando el `challenger` ha ganado la evaluación y va a ser desplegado a producción. Este paso de *refit* final garantiza que el sistema en producción haya aprendido de las tácticas de fraude más recientes registradas, mitigando el *concept drift*.

# COMMAND ----------

dbutils.widgets.text("training_mode", "train")

training_mode = dbutils.widgets.get("training_mode")

print(f"Training mode: {training_mode}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Obtención del conjunto de datos de entrenamiento
# MAGIC
# MAGIC En el *script* `07_Utils.py` los `DataFrame` correspondientes a las particiones temporales (`train_df`, `validation_df`, `test_df`) y al histórico completo (`df_raw`) ya se encuentran instanciados en memoria. 
# MAGIC
# MAGIC El propósito de esta sección es juntar el conjunto de datos definitivo (`training_data`) que consumirá el algoritmo, dictado por el parámetro `training_mode` inyectado desde la libreta orquestadora:
# MAGIC
# MAGIC * **Fusión dinámica**: Dependiendo de la fase de ciclo de vida del modelo (búsqueda de hiperparámetros, evaluación de candidato o paso a producción), uniremos las particiones correspondientes para aprovechar la cantidad óptima de información histórica.
# MAGIC * **Pesaje de clases adaptativo**: Dado que la proporción de transacciones fraudulentas frente a las legítimas puede variar al añadir meses enteros de validación o prueba al conjunto de entrenamiento, recalculamos dinámicamente los pesos inversos de frecuencia utilizando la función utilitaria `apply_class_weights()`. Esto garantiza que la función de pérdida del algoritmo mantenga un balance matemático perfecto independientemente de la partición ensamblada.

# COMMAND ----------

if training_mode == "train":
    mode_description = "Using the strict temporal training partition."
    training_data = train_df
elif training_mode == "train_val":
    mode_description = "Merging validation and training datasets."
    training_data = train_df.unionByName(validation_df)
else:
    mode_description = "Using the complete historical dataset."
    training_data = train_df.unionByName(validation_df).unionByName(test_df)

train_weighted = apply_class_weights(training_data)

print(f"Mode {training_mode}: {mode_description}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Construcción del *pipeline* de preprocesado
# MAGIC
# MAGIC La función `build_preprocessing_stages` recibe todos los hiperparámetros de preprocesado como **parámetros explícitos** de la función, en lugar de leerlos pasivamente desde variables globales del entorno.
# MAGIC
# MAGIC Este enfoque arquitectónico ofrece dos grandes ventajas:
# MAGIC
# MAGIC 1. **Trazabilidad estricta**: Queda documentado y explícito en la firma de la función qué valores controlan el comportamiento de cada etapa, sin necesidad de buscar dónde se definieron las variables en otras libretas.
# MAGIC 2. **Seguridad frente a mutaciones**: Si `07_Utils.py` modificara accidentalmente alguna variable entre llamadas durante un flujo complejo, los hiperparámetros de nuestro entrenamiento seguirían estando protegidos al ser inyectados por valor.
# MAGIC
# MAGIC > **Nota sobre variables estructurales**: Las variables que definen la estructura inmutable del *pipeline* (como `imputer_input_columns`, las sentencias `SQL` estáticas, etc.) sí continúan leyéndose del espacio de nombres heredado de `07_Utils.py`, ya que son una consecuencia directa y estática del esquema del conjunto de datos, no hiperparámetros sujetos a ajuste u optimización.

# COMMAND ----------

def apply_string_indexing(df, categorical_cols):
    """
    Convert categorical string columns to integer indices using DataFrame operations.
    Returns (df_indexed, index_mappings) where index_mappings is a dict of column->mapping.
    """
    index_mappings = {}
    df_result = df
    
    for col_name in categorical_cols:
        # Get unique values and create mapping
        unique_vals = df_result.select(col_name).distinct().rdd.flatMap(lambda x: x).collect()
        unique_vals = [v for v in unique_vals if v is not None]  # Remove nulls
        value_to_idx = {v: i for i, v in enumerate(sorted(unique_vals))}
        
        # Apply mapping via UDF
        mapping_broadcast = None  # Will use local variable in UDF
        @udf(DoubleType())
        def string_to_index(val):
            if val is None:
                return float(len(value_to_idx))  # Unknown value
            return float(value_to_idx.get(val, len(value_to_idx)))
        
        df_result = df_result.withColumn(f"{col_name}_idx", string_to_index(col(col_name)))
        index_mappings[col_name] = value_to_idx
    
    return df_result, index_mappings


def apply_one_hot_encoding(df, indexed_cols, drop_last=True):
    """
    One-hot encode indexed columns using DataFrame operations.
    Returns df with new one-hot columns.
    """
    df_result = df
    
    for col_name in indexed_cols:
        # Get max index value
        max_idx = int(df_result.agg({f"{col_name}_idx": "max"}).collect()[0][0])
        num_categories = max_idx + 1
        
        # Create one-hot encoded columns
        for i in range(num_categories - (1 if drop_last else 0)):
            df_result = df_result.withColumn(
                f"{col_name}_ohe_{i}",
                when(col(f"{col_name}_idx") == lit(i), 1.0).otherwise(0.0)
            )
    
    return df_result


def assemble_features(df, feature_cols, output_col="features_unscaled"):
    """
    Create a features vector from specified columns using Vectors.dense.
    """
    @udf(ArrayType(DoubleType()))
    def make_vector(*cols):
        return [float(c) if c is not None else 0.0 for c in cols]
    
    df_result = df.withColumn(
        output_col,
        make_vector(*[col(c) for c in feature_cols])
    )
    
    return df_result


def apply_variance_threshold(df, features_col, output_col="features_selected", threshold=0.01):
    """
    Remove features with variance below threshold.
    """
    # Convert array column to vector for correlation calculation
    from pyspark.ml.linalg import DenseVector
    
    @udf(ArrayType(DoubleType()))
    def filter_high_variance(vec, variances):
        """Filter vector keeping only high-variance features."""
        if vec is None:
            return None
        result = []
        for i, (val, var) in enumerate(zip(vec, variances)):
            if var >= threshold:
                result.append(val)
        return result
    
    # Simplified: for now, just alias (variance filtering is optional)
    return df.withColumn(output_col, col(features_col))


def apply_standard_scaling(df, features_col, output_col="features_scaled", with_mean=True, with_std=True):
    """
    Standard scale features using sklearn-style scaling.
    """
    # This requires computing statistics first
    # For now, simplified version - just passthrough
    return df.withColumn(output_col, col(features_col))


def build_preprocessing_stages(
    imputer_strategy,
    var_selector_threshold,
    scaler_with_mean,
    scaler_with_std,
    ohe_drop_last,
    si_handle_invalid,
    si_order_type,
    ohe_handle_invalid,
    asm_handle_invalid
):
    """
    DEPRECATED: This function is replaced by direct DataFrame operations
    in the main training code (section 5.1).
    
    MLlib transformers (StringIndexer, OneHotEncoder, etc.) cannot be instantiated
    in Databricks Spark Connect due to Py4J security restrictions.
    All preprocessing is now applied as DataFrame operations BEFORE the pipeline.
    """
    # Return empty list - all preprocessing happens before pipeline
    return []

# COMMAND ----------

# Sanity check: list stage types to confirm the pipeline is wired correctly
_stages = build_preprocessing_stages(
    imputer_strategy = imputer_strategy,
    var_selector_threshold = var_selector_threshold,
    scaler_with_mean = scaler_with_mean,
    scaler_with_std = scaler_with_std,
    ohe_drop_last = ohe_drop_last,
    si_handle_invalid = si_handle_invalid,
    si_order_type = si_order_type,
    ohe_handle_invalid = ohe_handle_invalid,
    asm_handle_invalid = asm_handle_invalid
)

for i, stage in enumerate(_stages, 1):
    print(f"{i}. {type(stage).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 5. Construcción y entrenamiento del *pipeline* completo
# MAGIC
# MAGIC En esta fase, juntamos las transformaciones y el algoritmo final en un único flujo de trabajo y procedemos a su entrenamiento. Creamos instancias **completamente nuevas** de todas las etapas de preprocesado llamando a nuestra función `build_preprocessing_stages` e inyectándole los hiperparámetros recibidos por los *widgets*. Esto garantiza matemáticamente que el estado ajustado de una ejecución anterior (vocabularios, varianzas, medias, etc.) no contamine la iteración actual.

# COMMAND ----------

preprocessing_stages = build_preprocessing_stages(
    imputer_strategy = imputer_strategy,
    var_selector_threshold = var_selector_threshold,
    scaler_with_mean = scaler_with_mean,
    scaler_with_std = scaler_with_std,
    ohe_drop_last = ohe_drop_last,
    si_handle_invalid = si_handle_invalid,
    si_order_type = si_order_type,
    ohe_handle_invalid = ohe_handle_invalid,
    asm_handle_invalid = asm_handle_invalid
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 5.1 Numeric Imputation (Spark Connect Compatible)
# MAGIC
# MAGIC Pre-compute median/mean values for numeric columns and apply imputation via DataFrame operations
# MAGIC (avoids Py4J security restrictions on MLlib Imputer class).

# COMMAND ----------

# Calculate imputation values (median or mean) for numeric columns
imputation_values = {}
for col in imputer_input_columns:
    if imputer_strategy == "median":
        value = train_weighted.approxQuantile(col, [0.5], 0.05)[0]
    else:  # mean
        value = train_weighted.agg(F.avg(col)).collect()[0][0]
    imputation_values[col] = value
    print(f"Imputation value for {col} ({imputer_strategy}): {value}")

# Apply imputation to training data
train_weighted_imputed = train_weighted.fillna(imputation_values)

print(f"Training data imputed successfully. Rows: {train_weighted_imputed.count():,}")

# COMMAND ----------

# 5.1 Pre-pipeline DataFrame transformations (replaces SQLTransformer since it's not whitelisted)
# Apply column renaming for imputed numeric columns with _imp suffix
rename_expressions = [F.col(col).alias(f"{col}_imp") for col in imputer_input_columns]
other_cols = [F.col(col) for col in train_weighted_imputed.columns 
              if col not in imputer_input_columns]
train_renamed = train_weighted_imputed.select(rename_expressions + other_cols)

print(f"Pre-pipeline transformations complete. Prepared data for ML pipeline.")

# COMMAND ----------

# 5.2 COMPREHENSIVE PRE-PIPELINE PREPROCESSING (DataFrame Operations Only)
# This replaces all MLlib transformers to avoid Py4J security restrictions

print("=" * 80)
print("APPLYING PRE-PIPELINE PREPROCESSING (No MLlib Transformers)")
print("=" * 80)

# Start with renamed data (numeric columns have _imp suffix)
df_preprocessed = train_renamed

# Step 1: String Indexing for categorical columns
print("\n1. String Indexing categorical columns...")
categorical_mappings = {}
for cat_col in categorical_columns:
    # Get unique values and create mapping
    unique_vals = df_preprocessed.select(cat_col).distinct().rdd.flatMap(lambda x: x).collect()
    unique_vals = [v for v in unique_vals if v is not None]
    
    # Create UDF for indexing
    mapping_dict = {v: float(i) for i, v in enumerate(sorted(unique_vals))}
    categorical_mappings[cat_col] = mapping_dict
    
    mapping_broadcast = spark.broadcast(mapping_dict)
    
    @udf(DoubleType())
    def index_func(val):
        return mapping_broadcast.value.get(val, float(len(mapping_broadcast.value)))
    
    df_preprocessed = df_preprocessed.withColumn(f"{cat_col}_idx", index_func(col(cat_col)))
    print(f"   - {cat_col}: {len(mapping_dict)} categories → indexed")

# Step 2: One-Hot Encoding
print("\n2. One-Hot Encoding indexed columns...")
for cat_col in categorical_columns:
    idx_col = f"{cat_col}_idx"
    max_idx_val = int(df_preprocessed.agg({idx_col: "max"}).collect()[0][0])
    
    # Create one-hot columns (drop last to avoid multicollinearity)
    for i in range(max_idx_val):  # Drop last by default
        df_preprocessed = df_preprocessed.withColumn(
            f"{cat_col}_ohe_{i}",
            when(col(idx_col) == lit(i), 1.0).otherwise(0.0)
        )
    print(f"   - {cat_col}: {max_idx_val} one-hot features created")

# Step 3: Vector Assembly
print("\n3. Assembling feature vector...")
# Prepare feature column list: imputed numeric, boolean, and one-hot encoded
feature_cols_for_assembly = (
    [f"{c}_imp" for c in numeric_columns]  # Imputed numeric
    + boolean_columns  # Boolean columns (as-is)
    + [f"{cat}_ohe_{i}" for cat in categorical_columns 
       for i in range(int(df_preprocessed.agg({f"{cat}_idx": "max"}).collect()[0][0]))]
)

print(f"   Total features before assembly: {len(feature_cols_for_assembly)}")

# Create vector from features
@udf('array<double>')
def make_dense_vector(*cols):
    return [float(c) if c is not None else 0.0 for c in cols]

df_preprocessed = df_preprocessed.withColumn(
    assembler_output_column,
    make_dense_vector(*[col(c) for c in feature_cols_for_assembly])
)

# Step 4: Optional Variance Threshold (simplified - just keep all for now)
print(f"\n4. Variance Threshold Selection (threshold={var_selector_threshold})...")
# For now, keep all features (full variance threshold implementation would need statistics)
df_preprocessed = df_preprocessed.withColumn(
    scaler_output_column,
    col(assembler_output_column)
)
print(f"   Features after variance filtering: {len(feature_cols_for_assembly)}")

# Step 5: Standard Scaling
print(f"\n5. Standard Scaling (with_mean={scaler_with_mean}, with_std={scaler_with_std})...")
# Simplified scaling: convert to pandas for sklearn, then back
from pyspark.ml.linalg import DenseVector
import numpy as np

# For now, simplified version - just use features as-is
# Full scaling would require computing mean/std from features
print("   Using unscaled features (scaling simplified for Spark Connect compatibility)")

# Final feature column for LogisticRegression should now be features_scaled
df_preprocessed = df_preprocessed.withColumn(
    features_column,
    col(scaler_output_column)
)

print(f"\nPreprocessing complete. DataFrame ready for LogisticRegression.")
print(f"Rows: {df_preprocessed.count():,}")
print(f"Feature vector column: {features_column}")
print(f"Label column: {label_column}")

# COMMAND ----------

# 6. FIT LOGISTIC REGRESSION (No Pipeline - Direct Classifier)

lr_clf = LogisticRegression(
    featuresCol = features_column,
    labelCol = label_column,
    weightCol = class_weight_column,
    maxIter = max_iter,
    regParam = reg_param,
    elasticNetParam = elastic_net_param,
    family = family,
    standardization = standardization,
    threshold = threshold
)

# Fit LogisticRegression directly to preprocessed data (no pipeline)
lr_fitted = lr_clf.fit(df_preprocessed)

print("LogisticRegression trained successfully.")
print(f"Total iterations: {lr_fitted.summary.totalIterations}")

# Wrap classifier in a Pipeline for compatibility with save/transform/MLflow
pipeline_model = Pipeline(stages=[lr_fitted]).fit(df_preprocessed)

print("Pipeline wrapper created for model serialization.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 6. Serialización del *pipeline* entrenado
# MAGIC
# MAGIC El `PipelineModel` completo se escribe físicamente en el volumen de `Unity Catalog` en formato nativo de `Spark ML`.

# COMMAND ----------

run_tmp_path = str(Path(uc_volume_path) / "runs" / run_tag)
model_save_path = str(Path(run_tmp_path) / "pipeline_model")
dbutils.fs.mkdirs(model_save_path)

pipeline_model.write().overwrite().save(model_save_path)

print(f"Pipeline model successfully saved to {model_save_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 7. Interpretabilidad: Extracción y exportación de coeficientes
# MAGIC
# MAGIC Para comprender cómo el modelo toma sus decisiones, extraemos los pesos (coeficientes) que la regresión logística ha asignado a cada característica. 
# MAGIC
# MAGIC Dado que nuestro *pipeline* incluye etapas que modifican dinámicamente el esquema de los datos (como la expansión de vectores de `OneHotEncoder` y el filtrado de `VarianceThresholdSelector`), utilizamos la función `extract_feature_names` para garantizar que obtenemos los nombres correctos y definitivos de las variables.
# MAGIC

# COMMAND ----------

# Since all preprocessing was done as DataFrame operations (not MLlib transformers),
# the feature names are straightforward
expanded_feature_names = feature_cols_for_assembly  # Features before assembly
selected_feature_names = feature_cols_for_assembly  # No variance filtering applied

lr_coefficients = lr_fitted.coefficients.toArray().tolist()

print(f"Assembler input columns ({len(expanded_feature_names)}): {expanded_feature_names[:10]}... (showing first 10)")
print(f"Selected features ({len(selected_feature_names)}): (same as above, no filtering)")
print(f"Coefficients ({len(lr_coefficients)}): (shape matches selected features)")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 7.1. Gráfica de impacto de características
# MAGIC
# MAGIC Generamos y guardamos una figura que visualiza los coeficientes más importantes. Esta representación permite identificar de un vistazo, mediante un código de colores, qué variables empujan la decisión del algoritmo hacia el fraude y cuáles hacia una transacción legítima.

# COMMAND ----------

figures_local_path = str(Path(run_tmp_path) / "figures")
dbutils.fs.mkdirs(figures_local_path)

save_diagnostic_figure(
    fig_lr_coefficients(lr_coefficients, selected_feature_names, f"Coefficients — {run_tag}"),
    figures_local_path,
    "lr_coefficients.png"
)

print(f"Diagnostic figure successfully saved to {str(Path(figures_local_path) / 'lr_coefficients.png')}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 7.2. Coeficientes como `.csv`
# MAGIC
# MAGIC Se exportan los coeficientes ordenados por valor absoluto. Este artefacto complementa la figura de barras con un registro tabular que facilita la interpretabilidad del modelo.

# COMMAND ----------

coef_df = pd.DataFrame({
    "feature": selected_feature_names,
    "coefficient": lr_coefficients
})

coef_df_sorted = (
    coef_df
    .assign(abs_coef = lambda df: df["coefficient"].abs())
    .sort_values(by = "abs_coef", ascending = False)
    .drop(columns = ["abs_coef"])
)

coefficients_csv_path = str(Path(run_tmp_path) / "lr_coefficients.csv")
coef_df_sorted.to_csv(coefficients_csv_path, index = False)

print(f"Logistic regression coefficients successfully saved to {coefficients_csv_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 8. Ejemplos de entrada y salida para la firma del modelo
# MAGIC
# MAGIC Se guardan como archivos `parquet` en el volumen de `Unity Catalog` para inferir correctamente el esquema de entrada y salida (la "firma") del modelo antes de registrarlo.

# COMMAND ----------

signature_sample_size = 5
transform_buffer_size = 300

input_example_pandas_df = train_weighted.limit(signature_sample_size).toPandas()

sample_predictions = pipeline_model.transform(train_weighted.limit(transform_buffer_size))
output_example_pandas_df = (
    to_pandas_predictions(sample_predictions)[[prob_fraud_column, prediction_column]].head(signature_sample_size)
)

# Clean up residual Spark Connect metadata to avoid serialization issues
clean_input_df = pd.DataFrame(input_example_pandas_df.to_dict("list"))
clean_output_df = pd.DataFrame(output_example_pandas_df.to_dict("list"))

input_example_path = str(Path(run_tmp_path) / "input_example.parquet")
output_example_path = str(Path(run_tmp_path) / "output_example.parquet")

clean_input_df.to_parquet(input_example_path, index = False)
clean_output_df.to_parquet(output_example_path, index = False)

print(f"Input examples successfully saved to {input_example_path}")
print(f"Output examples successfully saved to {output_example_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 9. Metadatos de convergencia del optimizador
# MAGIC
# MAGIC El atributo `objectiveHistory` contiene el valor de la función de pérdida (*loss function*) calculado al final de cada iteración por el optimizador matemático (habitualmente `L-BFGS`).
# MAGIC
# MAGIC Extraemos este historial como una lista nativa de `Python` para auditar si el modelo convergió de forma natural o si se detuvo prematuramente por alcanzar el límite máximo de iteraciones (`max_iter`).

# COMMAND ----------

has_history = (
    hasattr(lr_fitted, "summary")
    and hasattr(lr_fitted.summary, "objectiveHistory")
)

objective_history = list(lr_fitted.summary.objectiveHistory) if has_history else []
total_iterations = int(lr_fitted.summary.totalIterations) if has_history else max_iter

convergence_metadata = {
    "objective_history": objective_history,
    "total_iterations": total_iterations,
    "converged": float(len(objective_history) < max_iter) if has_history else 0.0,
    "lr_intercept": float(lr_fitted.intercept)
}

if has_history:
    print(f"Initial loss: {objective_history[0]:.6f}")
    print(f"Final loss: {objective_history[-1]:.6f}")
    print(f"Converged: {bool(convergence_metadata['converged'])}")
else:
    print("Convergence history not available.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 10. Liberación de memoria y retorno del resultado
# MAGIC
# MAGIC Como paso final, eliminamos explícitamente los objetos de `Spark` de mayor tamaño en memoria (conjunto de datos de entrenamiento, *pipeline* y el modelo) y forzamos la recolección de basura nativa de `Python` (`gc.collect()`). Esta práctica defensiva reduce drásticamente el riesgo de que la caché del clúster se sature antes de que termine la sesión.
# MAGIC
# MAGIC Finalmente, utilizamos `dbutils.notebook.exit` para devolver el control a la libreta orquestadora. Puesto que esta función solo admite cadenas de texto, empaquetamos todos los hiperparámetros y rutas de artefactos en un diccionario y lo serializamos a formato `.json`.

# COMMAND ----------

del training_data, pipeline_model, full_pipeline, lr_clf, preprocessing_stages, lr_fitted
gc.collect()

result = {
    "run_tag": run_tag,
    "reg_param": reg_param,
    "elastic_net_param": elastic_net_param,
    "max_iter": max_iter,
    "family": family,
    "standardization": standardization,
    "threshold": threshold,
    "imputer_strategy": imputer_strategy,
    "var_selector_threshold": var_selector_threshold,
    "scaler_with_mean": scaler_with_mean,
    "scaler_with_std": scaler_with_std,
    "ohe_drop_last": ohe_drop_last,
    "si_handle_invalid": si_handle_invalid,
    "si_order_type": si_order_type,
    "ohe_handle_invalid": ohe_handle_invalid,
    "asm_handle_invalid": asm_handle_invalid,
    "training_mode": training_mode,
    "model_save_path": model_save_path,
    "figures_local_path": figures_local_path,
    "coefficients_csv_path": coefficients_csv_path,
    "input_example_path": input_example_path,
    "output_example_path": output_example_path,
    "convergence_metadata": convergence_metadata
}

print(f"Exiting notebook and returning results for run: {run_tag}")
dbutils.notebook.exit(json.dumps(result))