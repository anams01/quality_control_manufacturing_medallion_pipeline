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

# Instancia del cliente de Feature Store
fe = FeatureEngineeringClient()

# Tabla spine
spine_table = src_table

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

print("Tablas estáticas creadas y constraints añadidas exitosamente.")

# Definir los FeatureLookup para cada tabla de características (ahora usando tablas estáticas)
machine_profile_lookup = FeatureLookup(
    table_name=machine_profile_table,
    feature_names=["machine_type", "installation_date", "machine_age_days", 
                   "nominal_cycle_time_s", "vibration_baseline_mm_s", 
                   "wear_rate_pct_month", "clean_room_class", "line_capacity_units_day"],
    lookup_key=["machine_id"]
)


# ===============================
# 2. CREAR TRAINING SET
# ===============================


# Crear el training set
training_set = fe.create_training_set(
    df=spark.table(spine_table),
    feature_lookups=[machine_profile_lookup],
    label="is_defective",
    exclude_columns=["label_available_date"],
)

# Materializa el DataFrame enriquecido
training_df = training_set.load_df()

# ===============================
# 2. COMPROBACIONES DE CALIDAD
# ===============================
print(f"Filas en spine: {spark.table(spine_table).count()}")
print(f"Filas en training_df: {training_df.count()}")

# Nulos por columna
import pyspark.sql.functions as F
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
        pass
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

# Create static copies of feature tables with primary keys
# (Materialized views and streaming tables cannot have PK constraints)
catalog = "workspace"
database = "ana_martin17"

# 1. Create static copy of gold_machine_profile (exclude line_id and ingestion_timestamp)
machine_profile_static = f"{catalog}.{database}.gold_machine_profile_static"
spark.sql(f"""
CREATE OR REPLACE TABLE {machine_profile_static} AS
SELECT 
    machine_id,
    machine_type,
    installation_date,
    machine_age_days,
    nominal_cycle_time_s,
    vibration_baseline_mm_s,
    wear_rate_pct_month,
    clean_room_class,
    line_capacity_units_day
FROM {catalog}.{database}.gold_machine_profile
""")
spark.sql(f"ALTER TABLE {machine_profile_static} ALTER COLUMN machine_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {machine_profile_static} DROP CONSTRAINT gold_machine_profile_static_pk")
except:
    pass
spark.sql(f"ALTER TABLE {machine_profile_static} ADD CONSTRAINT gold_machine_profile_static_pk PRIMARY KEY (machine_id)")

# 2. Create static copy of gold_supplier_profile (exclude ingestion_timestamp to avoid duplicates)
supplier_profile_static = f"{catalog}.{database}.gold_supplier_profile_static"
spark.sql(f"""
CREATE OR REPLACE TABLE {supplier_profile_static} AS
SELECT 
    supplier_id,
    supplier_name,
    country,
    onboarding_date,
    solder_thickness_mean_um,
    quality_rating,
    is_new_supplier
FROM {catalog}.{database}.gold_supplier_profile
""")
spark.sql(f"ALTER TABLE {supplier_profile_static} ALTER COLUMN supplier_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {supplier_profile_static} DROP CONSTRAINT gold_supplier_profile_static_pk")
except:
    pass
spark.sql(f"ALTER TABLE {supplier_profile_static} ADD CONSTRAINT gold_supplier_profile_static_pk PRIMARY KEY (supplier_id)")

# 3. Create static copy of gold_machine_agg_1h (PK only on machine_id for PIT joins, keep window_end for matching)
machine_agg_1h_static = f"{catalog}.{database}.gold_machine_agg_1h_static"
spark.sql(f"""
CREATE OR REPLACE TABLE {machine_agg_1h_static} AS
SELECT 
    machine_id,
    window_end,
    window_size AS window_size_1h,
    total_units AS total_units_1h,
    defects AS defects_1h,
    defect_rate AS defect_rate_1h,
    avg_vibration AS avg_vibration_1h,
    avg_tool_wear AS avg_tool_wear_1h,
    avg_temperature AS avg_temperature_1h,
    avg_solder_thickness AS avg_solder_thickness_1h,
    avg_alignment_error AS avg_alignment_error_1h
FROM {catalog}.{database}.gold_machine_agg_1h
""")
spark.sql(f"ALTER TABLE {machine_agg_1h_static} ALTER COLUMN machine_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {machine_agg_1h_static} DROP CONSTRAINT gold_machine_agg_1h_static_pk")
except:
    pass
spark.sql(f"ALTER TABLE {machine_agg_1h_static} ADD CONSTRAINT gold_machine_agg_1h_static_pk PRIMARY KEY (machine_id)")

# 4. For 24h, exclude window_end to avoid duplicate (only need one window_end column from 1h table)
machine_agg_24h_static = f"{catalog}.{database}.gold_machine_agg_24h_static"
spark.sql(f"""
CREATE OR REPLACE TABLE {machine_agg_24h_static} AS
SELECT 
    machine_id,
    window_size AS window_size_24h,
    total_units AS total_units_24h,
    defects AS defects_24h,
    defect_rate AS defect_rate_24h,
    avg_vibration AS avg_vibration_24h,
    avg_tool_wear AS avg_tool_wear_24h,
    avg_temperature AS avg_temperature_24h,
    avg_solder_thickness AS avg_solder_thickness_24h,
    avg_alignment_error AS avg_alignment_error_24h
FROM {catalog}.{database}.gold_machine_agg_24h
""")
spark.sql(f"ALTER TABLE {machine_agg_24h_static} ALTER COLUMN machine_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {machine_agg_24h_static} DROP CONSTRAINT gold_machine_agg_24h_static_pk")
except:
    pass
spark.sql(f"ALTER TABLE {machine_agg_24h_static} ADD CONSTRAINT gold_machine_agg_24h_static_pk PRIMARY KEY (machine_id)")

print("Static feature tables with primary keys created successfully.")

# Update feature lookups to use static tables
machine_profile_lookups = FeatureLookup(
    table_name=machine_profile_static,
    lookup_key="machine_id"
)

supplier_profile_lookups = FeatureLookup(
    table_name=supplier_profile_static,
    lookup_key="supplier_id"
)

machine_agg_1h_lookups = FeatureLookup(
    table_name=machine_agg_1h_static,
    lookup_key="machine_id"
)

machine_agg_24h_lookups = FeatureLookup(
    table_name=machine_agg_24h_static,
    lookup_key="machine_id"
)

feature_lookups_static = [
    machine_profile_lookups,
    supplier_profile_lookups,
    machine_agg_1h_lookups,
    machine_agg_24h_lookups
]

# Now create the training set
label = "is_defective"
exclude_columns = ["label_available_date"]

# Plan lógico
training_dataset = fe.create_training_set(
    df=spine_df,
    feature_lookups=feature_lookups_static,
    label=label,
    exclude_columns=exclude_columns
)

# Materialización (Ejecución real de los Joins distribuidos)
training_df = training_dataset.load_df()

print(f"Training dataset columns: {len(training_df.columns)}")
print("Training dataset created successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validación y Balance de Clases

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.sql import Window
from pyspark.sql.functions import sum as spark_sum

# Compute class balance directly without full count
print("Class balance (Defects):")
class_balance_df = (
    training_df.groupBy(label)
               .count()
               .withColumn("pct", round(col("count") / spark_sum("count").over(Window.partitionBy()) * 100, 2))
               .orderBy(label)
)

class_balance_rows = class_balance_df.collect()
total_count = sum(row['count'] for row in class_balance_rows)

print(f"Total rows: {total_count:,}\n")
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
