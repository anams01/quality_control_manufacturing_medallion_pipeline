# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Generación del conjunto de datos de entrenamiento
# MAGIC
# MAGIC **Autora**: Ana Martín Serrano
# MAGIC
# MAGIC El objetivo de esta libreta es construir el conjunto de datos estático para entrenar nuestro modelo predictivo de **detección de defectos en componentes electrónicos**.
# MAGIC
# MAGIC Combinaremos la **`gold_inspection_spine`** (donde residen las etiquetas `is_defective` de las inspecciones) con:
# MAGIC 1. `gold_machine_profile` (perfil estático de máquina)
# MAGIC 2. `gold_supplier_profile` (perfil estático de proveedor)
# MAGIC 3. `gold_machine_agg_1h` y `gold_machine_agg_24h` (métricas de degradación)
# MAGIC 4. Las variables de sensor capturadas en tiempo real se encuentran ya insertadas en la *spine*.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuración

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------


from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from datetime import datetime, timezone
from pyspark.sql.functions import col, count, max, round, when

# === BLOQUE PARA CREAR TABLA ESTÁTICA LISTA PARA FEATURE STORE ===
catalog = "workspace"
database = "ana_martin17"
src_table = f"{catalog}.{database}.gold_inspection_spine"
dst_table = f"{catalog}.{database}.gold_inspection_spine_static"

# Cargar la tabla streaming y filtrar nulos
df = spark.table(src_table).filter("unit_id IS NOT NULL")


# ===============================
# 1. GENERACIÓN DEL DATASET DE ENTRENAMIENTO CON JOIN PiT
# ===============================
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

# Instancia del cliente de Feature Store
fe = FeatureEngineeringClient()

# Tabla spine (ya estática y con PK)
spine_table = dst_table

# ===============================
# CONVERTIR VISTAS A TABLAS ESTÁTICAS CON CONSTRAINTS
# ===============================
# Las tablas de características son vistas y no pueden tener constraints.
# Convertimos cada vista a tabla estática.

# 1. Crear tabla estática de gold_machine_profile
machine_profile_table = f"{catalog}.{database}.gold_machine_profile_table"
spark.sql(f"""
CREATE OR REPLACE TABLE {machine_profile_table} AS
SELECT * FROM {catalog}.{database}.gold_machine_profile
""")

# Asegurar que machine_id es NOT NULL
spark.sql(f"""
ALTER TABLE {machine_profile_table}
ALTER COLUMN machine_id SET NOT NULL
""")

# Eliminar constraint si existe y crear nueva
try:
    spark.sql(f"ALTER TABLE {machine_profile_table} DROP CONSTRAINT gold_machine_profile_pk")
except:
    pass

spark.sql(f"""
ALTER TABLE {machine_profile_table}
ADD CONSTRAINT gold_machine_profile_pk PRIMARY KEY (machine_id)
""")

# 2. Crear tabla estática de gold_machine_agg_1h
agg_1h_table = f"{catalog}.{database}.gold_machine_agg_1h_table"
spark.sql(f"""
CREATE OR REPLACE TABLE {agg_1h_table} AS
SELECT * FROM {catalog}.{database}.gold_machine_agg_1h
""")

spark.sql(f"""
ALTER TABLE {agg_1h_table}
ALTER COLUMN machine_id SET NOT NULL
""")

spark.sql(f"""
ALTER TABLE {agg_1h_table}
ALTER COLUMN window_end SET NOT NULL
""")

try:
    spark.sql(f"ALTER TABLE {agg_1h_table} DROP CONSTRAINT gold_machine_agg_1h_pk")
except:
    pass

spark.sql(f"""
ALTER TABLE {agg_1h_table}
ADD CONSTRAINT gold_machine_agg_1h_pk PRIMARY KEY (machine_id, window_end)
""")

# 3. Crear tabla estática de gold_machine_agg_24h
agg_24h_table = f"{catalog}.{database}.gold_machine_agg_24h_table"
spark.sql(f"""
CREATE OR REPLACE TABLE {agg_24h_table} AS
SELECT * FROM {catalog}.{database}.gold_machine_agg_24h
""")

spark.sql(f"""
ALTER TABLE {agg_24h_table}
ALTER COLUMN machine_id SET NOT NULL
""")

spark.sql(f"""
ALTER TABLE {agg_24h_table}
ALTER COLUMN window_end SET NOT NULL
""")

try:
    spark.sql(f"ALTER TABLE {agg_24h_table} DROP CONSTRAINT gold_machine_agg_24h_pk")
except:
    pass

spark.sql(f"""
ALTER TABLE {agg_24h_table}
ADD CONSTRAINT gold_machine_agg_24h_pk PRIMARY KEY (machine_id, window_end)
""")

print("Tablas estáticas creadas y constraints añadidas exitosamente.")

# Definir los FeatureLookup para cada tabla de características (ahora usando tablas estáticas)
machine_profile_lookup = FeatureLookup(
    table_name=machine_profile_table,
    feature_names=None,  # Todas las columnas excepto la PK
    lookup_key=["machine_id"],
    timestamp_lookup_key=None  # No es temporal
)
agg_1h_lookup = FeatureLookup(
    table_name=agg_1h_table,
    feature_names=None,
    lookup_key=["machine_id"],
    timestamp_lookup_key="timestamp"  # Join temporal
)
agg_24h_lookup = FeatureLookup(
    table_name=agg_24h_table,
    feature_names=None,
    lookup_key=["machine_id"],
    timestamp_lookup_key="timestamp"
)


# ===============================
# 2. CREAR TRAINING SET CON JOIN PiT
# ===============================


# Crear el training set con join PiT
training_set = fe.create_training_set(
    df=spark.table(spine_table),
    feature_lookups=[machine_profile_lookup, agg_1h_lookup, agg_24h_lookup],
    label="is_defective",
    exclude_columns=["label_available_date"],  # Excluye metadatos no disponibles en inferencia
    # Si tienes más columnas a excluir, añádelas aquí
)

# Materializa el DataFrame enriquecido
training_df = training_set.load_df()

# ===============================
# 2. COMPROBACIONES DE CALIDAD
# ===============================
print(f"Filas en spine: {spark.table(spine_table).count()}")
print(f"Filas en training_df: {training_df.count()}")

# Nulos por columna
nulls = training_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in training_df.columns])
nulls.show(vertical=True, truncate=False)

# Balance de clases
training_df.groupBy("is_defective").count().show()




# ===============================
# 3. PERSISTENCIA DEL DATASET FINAL
# ===============================
final_table = f"{catalog}.{database}.gold_inspection_training_dataset"
training_df.write.mode("overwrite").format("delta").saveAsTable(final_table)

# Puedes añadir aquí la escritura de metadatos de trazabilidad si lo deseas

# Asegura que unit_id es NOT NULL y añade la clave primaria (solo si no existe)
spark.sql(f"""
ALTER TABLE {dst_table}
ALTER COLUMN unit_id SET NOT NULL
""")

# Elimina la constraint si ya existe, para evitar error de duplicado
try:
    spark.sql(f"ALTER TABLE {dst_table} DROP CONSTRAINT gold_inspection_spine_static_pk")
except Exception as e:
    if "does not exist" in str(e):
        pass  # No pasa nada si no existe
    elif "not found" in str(e):
        pass
    else:
        raise

spark.sql(f"""
ALTER TABLE {dst_table}
ADD CONSTRAINT gold_inspection_spine_static_pk PRIMARY KEY (unit_id)
""")

# COMMAND ----------

catalog = "workspace"
database = "ana_martin17"

gold_spine_table = f"{catalog}.{database}.gold_inspection_spine"
gold_machine_profile_table = f"{catalog}.{database}.gold_machine_profile"
gold_supplier_profile_table = f"{catalog}.{database}.gold_supplier_profile"
gold_machine_agg_1h_table = f"{catalog}.{database}.gold_machine_agg_1h"
gold_machine_agg_24h_table = f"{catalog}.{database}.gold_machine_agg_24h"

gold_training_dataset_table = f"{catalog}.{database}.gold_inspection_training_dataset"

fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Carga de la *spine*

# COMMAND ----------

spine_df = spark.table(gold_spine_table)
print(f"Spine rows: {spine_df.count():,}")
spine_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Lookups
# MAGIC
# MAGIC Para aplicar correctitud temporal Point-in-Time (PiT), usaremos el campo `timestamp` de la transacción.

# COMMAND ----------

timestamp_key = "timestamp"

# --- Perfil de Máquina ---
machine_profile_lookups = FeatureLookup(
    table_name=gold_machine_profile_table,
    lookup_key="machine_id"
    # Características estáticas no varían temporalmente normalmente, pero si
    # se tiene un campo TIMESERIES en origen, Databricks asociará el histórico de cambios.
)

# --- Perfil de Proveedor ---
supplier_profile_lookups = FeatureLookup(
    table_name=gold_supplier_profile_table,
    lookup_key="supplier_id"
)

# --- Agregaciones 1 Hora (máquina) ---
machine_agg_1h_lookups = FeatureLookup(
    table_name=gold_machine_agg_1h_table,
    lookup_key="machine_id",
    timestamp_lookup_key=timestamp_key
)

# --- Agregaciones 24 Horas (máquina) ---
machine_agg_24h_lookups = FeatureLookup(
    table_name=gold_machine_agg_24h_table,
    lookup_key="machine_id",
    timestamp_lookup_key=timestamp_key
)

feature_lookups = [
    machine_profile_lookups,
    supplier_profile_lookups,
    machine_agg_1h_lookups,
    machine_agg_24h_lookups
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Creación del Training Dataset
# MAGIC
# MAGIC Excluimos también `label_available_date` (si existe en el spine o de los joins) ya que es una variable post-inspección (*delayed feedback*) que no tendríamos en inferencia.

# COMMAND ----------

label = "is_defective"
exclude_columns = ["label_available_date"]

# Plan lógico
training_dataset = fe.create_training_set(
    df=spine_df,
    feature_lookups=feature_lookups,
    label=label,
    exclude_columns=exclude_columns
)

# Materialización (Ejecución real de los Joins distribuidos)
training_df = training_dataset.load_df()

print(f"Training dataset rows: {training_df.count():,}")
print(f"Training dataset columns: {len(training_df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validación y Balance de Clases

# COMMAND ----------

training_count = training_df.count()

print("Class balance (Defects):")
class_balance_rows = (
    training_df.groupBy(label)
               .count()
               .withColumn("pct", round(col("count") / training_count * 100, 2))
               .orderBy(label).collect()
)

for row in class_balance_rows:
    print(f"Defect {row[label]}: {row['count']:,} rows ({row['pct']}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persistencia y Trazabilidad

# COMMAND ----------

# Filtramos filas sin etiqueta
clean_training_df = training_df.filter("is_defective IS NOT NULL")

# Almacenado como tabla estática en Delta
(
    clean_training_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(gold_training_dataset_table)
)

print(f"Training dataset materializado y listo en: {gold_training_dataset_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusiones y Trazabilidad de MLflow
# MAGIC Una vez generado este dataset inmutable, cualquier corrida de entrenamiento de Modelado usará `gold_inspection_training_dataset`. Podremos viajar en el tiempo a versiones físicas anteriores para asegurar estricta reproducibilidad de nuestros modelos Random Forest / XGBoost ante una auditoría.
