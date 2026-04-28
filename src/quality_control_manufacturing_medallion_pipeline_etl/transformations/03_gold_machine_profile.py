"""
03_gold_machine_profile.py
===========================
Capa Oro — Perfiles estáticos de máquinas y proveedores.

Deriva características estáticas o de actualización lenta:
  - Perfil de máquina: tipo, línea, antigüedad, baseline de vibración
  - Perfil de proveedor: calidad histórica, grosor de soldadura esperado

Tabla generada:
  - gold_machine_profile : perfil enriquecido por machine_id
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F
import dlt
from pyspark.sql.functions import (
  col,
  expr
)

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"


@dlt.table(
  name="gold_machine_profile",
  comment="Static machine data and historical performance.",
  table_properties={
    "quality": "gold",
    "delta.enableChangeDataFeed": "true"
  }
)
def machine_profile():
  """
  Creates the gold_machine_profile table by enriching machine data with historical defect rates.
  This table serves as a feature view for the machine learning model.
  """
  machine_stream = dlt.read_stream("silver_machines")
  inspection_stream = dlt.read_stream("silver_inspections")

  # Calculate historical defect rates per machine
  machine_defect_rates = (
    inspection_stream
      .groupBy("machine_id")
      .agg(
        expr("count(*) as total_inspections"),
        expr("sum(case when is_defective = 1 then 1 else 0 end) as defective_count")
      )
      .withColumn("historical_defect_rate_pct", (col("defective_count") / col("total_inspections")) * 100)
  )

  # Join with static machine data
  return (
    machine_stream
      .join(machine_defect_rates, "machine_id", "left")
      .select(
        "machine_id",
        "machine_type",
        "machine_age_years",
        "last_maintenance_days",
        col("total_inspections").alias("historical_total_inspections"),
        col("historical_defect_rate_pct").alias("machine_historical_defect_rate_pct")
      )
  )

@dlt.table(
  name="gold_supplier_profile",
  comment="Static supplier data and historical performance.",
  table_properties={
    "quality": "gold",
    "delta.enableChangeDataFeed": "true"
  }
)
def supplier_profile():
  """
  Creates the gold_supplier_profile table by enriching supplier data with historical defect rates.
  This table serves as another feature view for the machine learning model.
  """
  supplier_stream = dlt.read_stream("silver_suppliers")
  inspection_stream = dlt.read_stream("silver_inspections")

  # Calculate historical defect rates per supplier
  supplier_defect_rates = (
    inspection_stream
      .groupBy("supplier_id")
      .agg(
        expr("count(*) as total_parts"),
        expr("sum(case when is_defective = 1 then 1 else 0 end) as defective_parts")
      )
      .withColumn("historical_defect_rate_pct", (col("defective_parts") / col("total_parts")) * 100)
  )

  # Join with static supplier data
  return (
    supplier_stream
      .join(supplier_defect_rates, "supplier_id", "left")
      .select(
        "supplier_id",
        "supplier_name",
        "part_type",
        col("historical_defect_rate_pct").alias("supplier_historical_defect_rate_pct")
      )
  )
