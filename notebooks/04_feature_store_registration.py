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
# MAGIC La arquitectura que publicaríamos es la siguiente:
# MAGIC
# MAGIC * **`gold_inspection_spine`**: esqueleto de entrenamiento. **No se publica** aquí porque no es una tabla de características.
# MAGIC * **`gold_machine_profile`**: Perfil estático de las máquinas. Se publica para recuperar información estática (edad, tipo) en tiempo real por `machine_id`.
# MAGIC * **`gold_supplier_profile`**: Perfil estático de los proveedores. Se publica para recuperar histórico de calidad por `supplier_id`.
# MAGIC * **`gold_machine_agg_1h` y `gold_machine_agg_24h`**: Agregaciones dinámicas de comportamiento. Se publican para consultar el estado operativo de la máquina en la última hora y día.

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

catalog = "workspace"
database = "ana_martin17"  # Schema del proyecto en Unity Catalog

# Tablas origen en Delta
gold_machine_profile_table = f"{catalog}.{database}.gold_machine_profile"
gold_supplier_profile_table = f"{catalog}.{database}.gold_supplier_profile"
gold_machine_agg_1h_table = f"{catalog}.{database}.gold_machine_agg_1h"
gold_machine_agg_24h_table = f"{catalog}.{database}.gold_machine_agg_24h"

# Un único Online Store para todo el proyecto
online_store_name = "quality_control_online_store"

# Nombres de las tablas una vez publicadas
online_machine_profile_table = f"{catalog}.{database}.online_machine_profile"
online_supplier_profile_table = f"{catalog}.{database}.online_supplier_profile"
online_machine_agg_1h_table = f"{catalog}.{database}.online_machine_agg_1h"
online_machine_agg_24h_table = f"{catalog}.{database}.online_machine_agg_24h"

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
# MAGIC Las agregaciones de 1h y 24h se generan probablemente usando ventanas deslizantes o agregaciones batch escalonadas, por lo que el modo de publicación adecuado es `TRIGGERED`. Propagará de forma incremental los cambios en las tablas originarias cada vez que finalice el pipeline DLT.

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

# Publicar agregaciones de 1 hora (Lookup por machine_id y timestamp)
# fe.publish_table(
#     online_store=online_store_name,
#     source_table_name=gold_machine_agg_1h_table,
#     online_table_name=online_machine_agg_1h_table,
#     publish_mode=publish_mode
# )

# Publicar agregaciones de 24 horas
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
