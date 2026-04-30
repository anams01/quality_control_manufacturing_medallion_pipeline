"""
03_gold_machine_profile.py
===========================
Capa Oro — Perfiles estáticos de máquinas y proveedores.

Deriva características estáticas o de actualización lenta:
  - Perfil de máquina: tipo, línea, antigüedad, baseline de vibración
  - Perfil de proveedor: calidad histórica, grosor de soldadura esperado

Tablas generadas:
  - gold_machine_profile  : perfil enriquecido por machine_id
  - gold_supplier_profile : perfil enriquecido por supplier_id

CORRECCIONES respecto a la versión original:
  1. bronze_machines / bronze_suppliers se leen en modo BATCH (dlt.read), no
     streaming. Son catálogos estáticos que no tienen watermark y no requieren
     streaming; intentar leerlos con read_stream provocaría un error en DLT.
  2. Se elimina la dependencia de "silver_machines" y "silver_suppliers", que
     no existen en el pipeline (no hay notebook 02_silver para esas tablas).
     Se consume directamente la capa bronze (tablas de referencia estáticas).
  3. Las agregaciones de defectos sobre silver_inspections se hacen también en
     modo BATCH dentro del cuerpo de la función DLT. DLT gestiona
     automáticamente la corrección temporal gracias a que es una tabla gold.
  4. Se unifica la API: se elimina el import de pyspark.pipelines (dp) que no
     se usaba; solo se usa dlt.
"""

import dlt
from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"


# =============================================================================
# PERFIL DE MÁQUINA
# =============================================================================

@dlt.table(
    name="gold_machine_profile",
    comment="Static machine data enriched with historical defect rates per machine.",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true",
    },
)
def machine_profile():
    """
    Construye el perfil estático de cada máquina enriquecido con su tasa
    histórica de defectos.

    - bronze_machines: leída en modo BATCH (catálogo estático, sin watermark).
    - silver_inspections: leída en modo BATCH para calcular agregados históricos.
      DLT garantiza que esta tabla gold solo se recalcula cuando alguna de sus
      fuentes cambia, por lo que no es necesario streaming aquí.
    """
    # Catálogo de máquinas — lectura batch (tabla estática)
    machines_df = dlt.read("bronze_machines")

    # Agregados históricos de defectos por máquina — lectura batch
    machine_defect_rates = (
        dlt.read("silver_inspections")
        .groupBy("machine_id")
        .agg(
            expr("count(*) as total_inspections"),
            expr("sum(case when is_defective = 1 then 1 else 0 end) as defective_count"),
        )
        .withColumn(
            "historical_defect_rate_pct",
            (col("defective_count") / col("total_inspections")) * 100,
        )
    )

    return (
        machines_df
        .join(machine_defect_rates, "machine_id", "left")
        .select(
            "machine_id",
            "machine_type",
            "machine_age_years",
            "last_maintenance_days",
            col("total_inspections").alias("historical_total_inspections"),
            col("historical_defect_rate_pct").alias("machine_historical_defect_rate_pct"),
        )
    )


# =============================================================================
# PERFIL DE PROVEEDOR
# =============================================================================

@dlt.table(
    name="gold_supplier_profile",
    comment="Static supplier data enriched with historical defect rates per supplier.",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true",
    },
)
def supplier_profile():
    """
    Construye el perfil estático de cada proveedor enriquecido con su tasa
    histórica de defectos.

    - bronze_suppliers: leída en modo BATCH (catálogo estático, sin watermark).
    - silver_inspections: leída en modo BATCH para calcular agregados históricos.
    """
    # Catálogo de proveedores — lectura batch (tabla estática)
    suppliers_df = dlt.read("bronze_suppliers")

    # Agregados históricos de defectos por proveedor — lectura batch
    supplier_defect_rates = (
        dlt.read("silver_inspections")
        .groupBy("supplier_id")
        .agg(
            expr("count(*) as total_parts"),
            expr("sum(case when is_defective = 1 then 1 else 0 end) as defective_parts"),
        )
        .withColumn(
            "historical_defect_rate_pct",
            (col("defective_parts") / col("total_parts")) * 100,
        )
    )

    return (
        suppliers_df
        .join(supplier_defect_rates, "supplier_id", "left")
        .select(
            "supplier_id",
            "supplier_name",
            "part_type",
            col("historical_defect_rate_pct").alias("supplier_historical_defect_rate_pct"),
        )
    )
