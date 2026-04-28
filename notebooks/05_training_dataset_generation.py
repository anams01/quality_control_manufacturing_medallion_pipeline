# Databricks notebook source
# MAGIC %md
# MAGIC # Generación del Conjunto de Datos de Entrenamiento con Feature Store
# MAGIC
# MAGIC **Autora**: Ana Martín Serrano
# MAGIC
# MAGIC Este notebook utiliza el Databricks Feature Store para construir el conjunto de datos de entrenamiento. Este enfoque reemplaza los costosos joins manuales por `FeatureLookups` que realizan uniones `point-in-time` de manera automática y eficiente.
# MAGIC
# MAGIC El proceso es el siguiente:
# MAGIC 1. Se define una **tabla base (spine)** que contiene los eventos de inspección (`inspection_id`), la marca de tiempo (`timestamp`) y la etiqueta a predecir (`is_defective`).
# MAGIC 2. Se definen **`FeatureLookups`** que apuntan a nuestras tablas de características en la capa Gold.
# MAGIC 3. Se invoca a `fe.create_training_set`, que se encarga de unir las características correctas a cada inspección basándose en la marca de tiempo.
# MAGIC 4. El conjunto de datos resultante se materializa en una tabla Delta final, lista para el entrenamiento del modelo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración e Instalación

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.1.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Definición de Parámetros
# MAGIC
# MAGIC Se definen los nombres del catálogo, la base de datos y las tablas que se utilizarán.

# COMMAND ----------

catalog = "workspace"
database = "ana_martin17"

# --- Tablas de Origen (Capa Gold) ---
spine_table_name = f"{catalog}.{database}.gold_inspection_spine"
machine_profile_table_name = f"{catalog}.{database}.gold_machine_profile"
supplier_profile_table_name = f"{catalog}.{database}.gold_supplier_profile"
machine_aggregations_table_name = f"{catalog}.{database}.gold_machine_aggregations"

# --- Tabla de Destino ---
training_dataset_table_name = f"{catalog}.{database}.gold_inspection_training_dataset"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Creación del Conjunto de Entrenamiento
# MAGIC
# MAGIC Se inicializa el cliente del `Feature Store` y se definen los `lookups` para cada tabla de características.

# COMMAND ----------

fe = FeatureEngineeringClient()

# 1. Feature Lookup para el perfil estático de la máquina
machine_profile_lookup = FeatureLookup(
    table_name=machine_profile_table_name,
    lookup_key="machine_id"
)

# 2. Feature Lookup para el perfil estático del proveedor
supplier_profile_lookup = FeatureLookup(
    table_name=supplier_profile_table_name,
    lookup_key="supplier_id"
)

# 3. Feature Lookup para las agregaciones dinámicas de la máquina (Point-in-Time)
#    Se utiliza `timestamp_lookup_key` para asegurar la correctitud temporal.
machine_aggregations_lookup = FeatureLookup(
    table_name=machine_aggregations_table_name,
    lookup_key="machine_id",
    timestamp_lookup_key="timestamp"
)

# Lista de todos los lookups
feature_lookups = [
    machine_profile_lookup,
    supplier_profile_lookup,
    machine_aggregations_lookup
]

# Cargar la tabla base (spine)
spine_df = spark.table(spine_table_name)

# Crear el conjunto de entrenamiento
# El Feature Store se encarga de los joins Point-in-Time
training_set = fe.create_training_set(
    df=spine_df,
    feature_lookups=feature_lookups,
    label="is_defective",
    exclude_columns=["inspection_id"] # Excluimos el ID de inspección para no sobreajustar
)

# Materializar el DataFrame con todas las características
training_df = training_set.load_df()

print("Conjunto de datos de entrenamiento creado con éxito.")
training_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Validación y Persistencia
# MAGIC
# MAGIC Se realizan comprobaciones de calidad sobre el `DataFrame` resultante y se guarda como una tabla Delta final.

# COMMAND ----------

# DBTITLE 1,Validación de Nulos y Conteo
print(f"Número de filas en el conjunto de entrenamiento: {training_df.count():,}")
print(f"Número de columnas: {len(training_df.columns)}")

# Comprobar nulos en columnas clave
training_df.select([col(c).isNull().alias(c) for c in training_df.columns]).display()

# Balance de clases
print("Balance de clases:")
training_df.groupBy("is_defective").count().display()

# COMMAND ----------

# DBTITLE 1,Guardar el Conjunto de Datos Final
# Aumentar el timeout para la materialización de datasets grandes
spark.conf.set("spark.databricks.execution.timeout", "3600") # 1 hora

print(f"Guardando el conjunto de datos en la tabla: {training_dataset_table_name}")

(
    training_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(training_dataset_table_name)
)

print("¡El conjunto de datos de entrenamiento ha sido guardado con éxito!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Conclusión
# MAGIC
# MAGIC El conjunto de datos `gold_inspection_training_dataset` está ahora listo y materializado. Contiene todas las características necesarias, unidas con correctitud "point-in-time", y puede ser utilizado directamente por los notebooks de entrenamiento de modelos (`07_...` y `08_...`).
# MAGIC
# MAGIC Este enfoque no solo es más eficiente, sino también más robusto y fácil de mantener.
