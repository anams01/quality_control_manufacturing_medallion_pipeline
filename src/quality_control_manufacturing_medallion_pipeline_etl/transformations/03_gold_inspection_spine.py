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
