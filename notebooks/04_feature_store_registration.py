# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Publicación en el `Online Feature Store`
# MAGIC
# MAGIC **Autora**: Ana Martín Serrano
# MAGIC
# MAGIC El objetivo de esta libreta es mostrar cómo publicar las tablas de la capa `Gold` de nuestro proyecto de **Control de Calidad en Manufactura** en un `Online Feature Store` respaldado por `Lakebase` para inferencia en tiempo real.
# MAGIC
# MAGIC Esta libreta **no contiene lógica de transformación propia**. Es un *script* de configuración que registra las tablas en el `Online Feature Store` y lanza la sincronización.
# MAGIC
# MAGIC La arquitectura que publicamos es la siguiente:
# MAGIC
# MAGIC * **`gold_inspection_spine`**: esqueleto de entrenamiento. **No se publica** aquí porque no es una tabla de características.
# MAGIC * **`gold_machine_profile`**: Perfil estático de las máquinas. Se publica para recuperar información estática (edad, tipo) en tiempo real por `machine_id`.
# MAGIC * **`gold_supplier_profile`**: Perfil estático de los proveedores. Se publica para recuperar histórico de calidad por `supplier_id`.
# MAGIC * **`gold_machine_agg_1h`** y **`gold_machine_agg_24h`**: Agregaciones dinámicas de comportamiento por máquina en ventana de 1 h y 24 h respectivamente. Generadas por `03_gold_machine_aggregations.py`.
# MAGIC
# MAGIC > **Nota**: Las tablas `gold_machine_agg_7d` y `gold_machine_agg_30d` existen en el pipeline DLT
# MAGIC > pero no se publican en el Online Store porque su latencia de actualización (días) no aporta
# MAGIC > valor en inferencia en tiempo real. Pueden consultarse directamente desde Delta si se necesitan
# MAGIC > en procesos batch.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1. Importación de librerías y configuración

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.13.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# COMMAND ----------

catalog  = "workspace"
database = "ana_martin17"   # Schema del proyecto en Unity Catalog

# --- Tablas origen en Delta (generadas por el pipeline DLT de la capa Gold) ---
# Perfiles estáticos
gold_machine_profile_table  = f"{catalog}.{database}.gold_machine_profile"
gold_supplier_profile_table = f"{catalog}.{database}.gold_supplier_profile"

# Agregaciones dinámicas — nombres correctos según 03_gold_machine_aggregations.py
# FIX: la versión original referenciaba gold_machine_agg_1h / gold_machine_agg_24h
# directamente, pero la tabla generada se llamaba gold_machine_aggregations (una sola).
# Tras la corrección del pipeline DLT, ahora sí existen las 4 tablas separadas.
gold_machine_agg_1h_table  = f"{catalog}.{database}.gold_machine_agg_1h"
gold_machine_agg_24h_table = f"{catalog}.{database}.gold_machine_agg_24h"

# Un único Online Store para todo el proyecto
online_store_name = "quality_control_online_store"

# Nombres de las tablas una vez publicadas en el Online Store
online_machine_profile_table  = f"{catalog}.{database}.online_machine_profile"
online_supplier_profile_table = f"{catalog}.{database}.online_supplier_profile"
online_machine_agg_1h_table   = f"{catalog}.{database}.online_machine_agg_1h"
online_machine_agg_24h_table  = f"{catalog}.{database}.online_machine_agg_24h"

# Instanciar el cliente
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2. Creación del `Online Feature Store` (Comentado)
# MAGIC
# MAGIC (Código ilustrativo de cómo se crearía el `Online Feature Store` si estuviese habilitado).

# COMMAND ----------

# capacity = "CU_1"
#
# try:
#     fe.create_online_store(name=online_store_name, capacity=capacity)
#     print(f"Online store created: {online_store_name}")
# except Exception as e:
#     if "already exists" in str(e).lower():
#         print(f"Skipping creation: {online_store_name}")
#     else:
#         raise

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3. Publicación de las tablas
# MAGIC
# MAGIC Las agregaciones de 1 h y 24 h se actualizan en cada ejecución del pipeline DLT, por lo que
# MAGIC el modo de publicación adecuado es `TRIGGERED`. Esto propagará de forma incremental los cambios
# MAGIC cada vez que finalice el pipeline DLT.

# COMMAND ----------

publish_mode = "TRIGGERED"

# Publicar perfil de máquinas (Lookup por machine_id)
# fe.publish_table(
#     online_store=online_store_name,
#     source_table_name=gold_machine_profile_table,
#     online_table_name=online_machine_profile_table,
#     publish_mode=publish_mode
# )

# Publicar perfil de proveedores (Lookup por supplier_id)
# fe.publish_table(
#     online_store=online_store_name,
#     source_table_name=gold_supplier_profile_table,
#     online_table_name=online_supplier_profile_table,
#     publish_mode=publish_mode
# )

# Publicar agregaciones de 1 hora (Lookup por machine_id + window_start)
# fe.publish_table(
#     online_store=online_store_name,
#     source_table_name=gold_machine_agg_1h_table,
#     online_table_name=online_machine_agg_1h_table,
#     publish_mode=publish_mode
# )

# Publicar agregaciones de 24 horas (Lookup por machine_id + window_start)
# fe.publish_table(
#     online_store=online_store_name,
#     source_table_name=gold_machine_agg_24h_table,
#     online_table_name=online_machine_agg_24h_table,
#     publish_mode=publish_mode
# )

print("Explicación de publicación completa.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4. Conclusiones
# MAGIC
# MAGIC - Las características calculadas en la capa `Gold` quedan listas para servirse a latencia de ms en el pipeline de inferencia.
# MAGIC - Durante el deployment, el modelo solo necesitará consultar las características usando los identificadores (`machine_id`, `supplier_id`), omitiendo el tedioso proceso de uniones manuales.
# MAGIC - Las tablas de agregación (`gold_machine_agg_1h`, `gold_machine_agg_24h`) usan `window_start` como clave temporal además de `machine_id`; el Feature Store realizará el lookup point-in-time correctamente gracias al campo `timestamp_lookup_key` configurado en `05_training_dataset_generation.py`.