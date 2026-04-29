"""
03_gold_machine_aggregations.py
================================
Capa Oro — Agregaciones dinámicas de comportamiento por máquina.

En lugar de hacer JOIN entre vistas streaming (no soportado en Spark),
se crean 4 tablas independientes, una por ventana temporal.

Tablas generadas:
  - gold_machine_agg_1h   : métricas por máquina en ventana de 1 hora
  - gold_machine_agg_24h  : métricas por máquina en ventana de 24 horas
  - gold_machine_agg_7d   : métricas por máquina en ventana de 7 días
  - gold_machine_agg_30d  : métricas por máquina en ventana de 30 días
"""

import dlt
from pyspark.sql import Window
from pyspark.sql.functions import (
  col,
  expr,
  avg,
  sum as _sum,
  count
)

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"
WATERMARK     = "30 days"



@dlt.table(
  name="gold_machine_aggregations",
  comment="""
    Aggregated machine sensor readings and defect rates over various time windows.
    This table is a feature view used for point-in-time lookups.
  """,
  table_properties={
    "quality": "gold",
    "delta.enableChangeDataFeed": "true"
  }
)
def machine_aggregations():
  """
  Calculates rolling window aggregations for machine sensor data and defect rates.
  This approach is highly efficient as it calculates all aggregations in a single pass
  over the data, avoiding the creation of large, intermediate tables.
  """
  
  inspections_df = dlt.read_stream("silver_inspections")

  # Define the time windows in hours and their string representations
  windows = {
    1: "1h",
    24: "24h",
    7*24: "7d",
    30*24: "30d"
  }

  # Define all sensor columns to aggregate
  sensor_columns = [
    "temperature_celsius", "pressure_bar", "vibration_mm_s", "voltage_v", "current_ma",
    "humidity_pct", "particle_count_m3", "solder_thickness_um", "alignment_error_um",
    "optical_density", "tool_wear_pct", "time_since_maintenance_h", "production_speed_pct",
    "operator_experience_yrs", "cycle_time_s"
  ]

  # Iteratively add aggregated columns for each window
  agg_df = inspections_df
  for hours, name in windows.items():
    window_spec = (
      Window
        .partitionBy("machine_id")
        .orderBy(col("timestamp").cast("long"))
        .rangeBetween(-hours * 3600, 0) # Window is from X hours ago to current event
    )
    
    # Calculate aggregations for all sensor columns
    for sensor_col in sensor_columns:
      agg_df = agg_df.withColumn(f"avg_{sensor_col}_{name}", avg(sensor_col).over(window_spec))
    
    # Calculate defect rate over the window
    defective_sum = _sum(col("is_defective")).over(window_spec)
    total_count = count(col("is_defective")).over(window_spec)
    agg_df = agg_df.withColumn(f"defect_rate_{name}", (defective_sum / total_count) * 100)

  # Select the final columns for the feature table
  # The primary keys are machine_id and timestamp, which are required for the Feature Store
  base_columns = ["machine_id", "timestamp", "inspection_id"]
  
  # Add all aggregated columns
  aggregated_columns = []
  for name in windows.values():
    for sensor_col in sensor_columns:
      aggregated_columns.append(f"avg_{sensor_col}_{name}")
    aggregated_columns.append(f"defect_rate_{name}")
  
  final_columns = base_columns + aggregated_columns
  
  return agg_df.select(*final_columns)