"""
03_gold_inspection_spine.py
============================
Capa Oro — Tabla spine (ancla) para entrenamiento del modelo ML.

Contiene exclusivamente:
  - Identificadores primarios (unit_id)
  - Timestamp exacto del evento
  - Variable objetivo (is_defective)
  - Features que llegan inherentemente con la inspección en tiempo real

NO se unen aquí perfiles ni agregaciones — eso lo hace el feature store
automáticamente en la fase de modelado (point-in-time correctness).

Tabla generada:
  - gold_inspection_spine : tabla base para entrenamiento e inferencia
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F
import dlt
from pyspark.sql.functions import col

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"
WATERMARK     = "30 days"


@dp.table(
    name="gold_inspection_spine",
    comment=(
        "Tabla spine para entrenamiento del modelo de detección de defectos. "
        "Contiene unit_id, timestamp, variable objetivo (is_defective) y "
        "features en tiempo real disponibles en el momento de la inspección. "
        "Los perfiles de máquina y agregaciones se inyectan via feature store."
    ),
    table_properties={"delta.enableChangeDataFeed": "true"},
)
def gold_inspection_spine():
    return (
        spark.readStream
        .table(f"{CATALOG}.{SCHEMA_TABLES}.silver_inspections_labeled")
        .withWatermark("timestamp", WATERMARK)
        # Solo registros con etiqueta disponible (histórico de entrenamiento)
        .filter(F.col("is_defective").isNotNull())
        .select(
            # --- Identificadores ---
            "unit_id",
            "timestamp",

            # --- Variable objetivo ---
            F.col("is_defective").cast("integer").alias("is_defective"),

            # --- Features en tiempo real (disponibles en inferencia) ---
            "machine_id",
            "line_id",
            "shift",
            "supplier_id",
            "material_batch_id",

            # Sensores físicos
            "temperature_celsius",
            "pressure_bar",
            "vibration_mm_s",
            "voltage_v",
            "current_ma",
            "humidity_pct",
            "particle_count_m3",
            "solder_thickness_um",
            "alignment_error_um",
            "optical_density",

            # Parámetros de proceso
            "tool_wear_pct",
            "time_since_maintenance_h",
            "production_speed_pct",
            "operator_experience_yrs",
            "cycle_time_s",

            # Metadatos
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )

@dlt.table(
  name="gold_inspection_spine",
  comment="""
    The spine table for the training dataset. 
    Contains the primary key, timestamp, and the label for each inspection event.
    This table forms the base for creating the training set, where features will be joined 
    point-in-time.
  """,
  table_properties={
    "quality": "gold"
  }
)
def inspection_spine():
  """
  Creates the spine table containing the core information for each inspection:
  - inspection_id: The primary key for the event.
  - timestamp: The event timestamp, crucial for point-in-time joins.
  - is_defective: The target variable (label) to be predicted.
  - machine_id: Foreign key for joining machine features.
  - supplier_id: Foreign key for joining supplier features.
  """
  return (
    dlt.read_stream("silver_inspections")
      .select(
        "inspection_id",
        "timestamp",
        "is_defective",
        "machine_id",
        "supplier_id"
      )
  )
