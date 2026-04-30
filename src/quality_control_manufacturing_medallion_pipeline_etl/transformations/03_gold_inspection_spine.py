"""
03_gold_inspection_spine.py
============================
Capa Oro — Tabla spine (ancla) para entrenamiento del modelo ML.

Contiene exclusivamente:
  - Identificadores primarios (unit_id)
  - Timestamp exacto del evento
  - Variable objetivo (is_defective)
  - Claves foráneas para el Feature Store (machine_id, supplier_id)

NO se unen aquí perfiles ni agregaciones — eso lo hace el feature store
automáticamente en la fase de modelado (point-in-time correctness).

Tabla generada:
  - gold_inspection_spine : tabla base para entrenamiento e inferencia

CORRECCIONES respecto a la versión original:
  1. inspection_id → unit_id. El esquema real de bronze/silver usa unit_id
     como clave primaria; inspection_id no existe en ninguna tabla del pipeline.
  2. Fuente cambiada de silver_inspections a silver_inspections_labeled.
     silver_inspections no contiene is_defective (esa columna solo existe tras
     el stream-stream join con silver_labels en 02_silver_transformation.py).
  3. Se elimina el import de pyspark.pipelines (dp) que no se usaba.
     Se unifica la API usando exclusivamente dlt.
"""

import dlt
from pyspark.sql import functions as F

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"


@dlt.table(
    name="gold_inspection_spine",
    comment="""
        Spine table for the ML training dataset.
        Contains the primary key (unit_id), timestamp, label (is_defective)
        and the foreign keys needed for Feature Store point-in-time lookups.
        Profiles and aggregations are NOT joined here — the Feature Store
        injects them automatically during training/inference.
    """,
    table_properties={
        "quality": "gold",
    },
)
def inspection_spine():
    """
    Construye la tabla spine leyendo desde silver_inspections_labeled, que es
    la única tabla de la capa plata que contiene is_defective (resultado del
    stream-stream join con silver_labels).

    Columnas seleccionadas:
      - unit_id      : clave primaria del evento de inspección.
      - timestamp    : instante temporal, clave para los joins PiT del Feature Store.
      - is_defective : variable objetivo (puede ser null por delayed feedback).
      - machine_id   : clave foránea → gold_machine_profile / gold_machine_agg_*.
      - supplier_id  : clave foránea → gold_supplier_profile.
    """
    return (
        dlt.read_stream("silver_inspections_labeled")
        .select(
            "unit_id",
            "timestamp",
            "is_defective",
            "machine_id",
            "supplier_id",
        )
    )
