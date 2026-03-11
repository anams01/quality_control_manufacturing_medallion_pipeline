"""
02_silver_transformation.py
============================
Capa Plata — Limpieza, validación, cuarentena y enriquecimiento.

Tablas generadas:
  - silver_inspections_quarantine  : registros que no superan las reglas (DLQ)
  - silver_labels_quarantine       : etiquetas anómalas (DLQ)
  - silver_inspections             : inspecciones limpias
  - silver_labels                  : etiquetas limpias
  - silver_inspections_labeled     : tabla de hechos unificada (stream-stream join)
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F
from rules import get_rules_by_tag

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CATALOG  = "workspace"
SCHEMA   = "quality_control_manufacturing"
VOLUME   = "landing_zone"

BASE_PATH       = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
CHECKPOINT_BASE = f"{BASE_PATH}/_checkpoints"

WATERMARK_DELAY = "30 days"


# =============================================================================
# HELPERS
# =============================================================================

def _build_quarantine_flag(rules: dict) -> F.Column:
    all_valid = F.lit(True)
    for rule in rules.values():
        all_valid = all_valid & F.expr(rule["constraint"])
    return (~all_valid).alias("is_quarantined")


INSPECTION_RULES = get_rules_by_tag("inspections")
LABEL_RULES      = get_rules_by_tag("labels")


# =============================================================================
# INSPECCIONES — Cuarentena
# =============================================================================

@dp.table(
    name="silver_inspections_quarantine",
    comment="Registros de inspección que no superan las reglas de calidad (DLQ).",
)
def silver_inspections_quarantine():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA}.bronze_inspections")
        .withColumn("is_quarantined", _build_quarantine_flag(INSPECTION_RULES))
        .filter(F.col("is_quarantined") == True)
        .drop("is_quarantined")
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================================
# INSPECCIONES — Tabla limpia
# =============================================================================

@dp.table(
    name="silver_inspections",
    comment="Inspecciones limpias y validadas.",
)
def silver_inspections():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA}.bronze_inspections")
        .withColumn("is_quarantined", _build_quarantine_flag(INSPECTION_RULES))
        .filter(F.col("is_quarantined") == False)
        .drop("is_quarantined")
    )


# =============================================================================
# ETIQUETAS — Cuarentena
# =============================================================================

@dp.table(
    name="silver_labels_quarantine",
    comment="Etiquetas de defecto que no superan las reglas de calidad (DLQ).",
)
def silver_labels_quarantine():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA}.bronze_labels")
        .withColumn("is_quarantined", _build_quarantine_flag(LABEL_RULES))
        .filter(F.col("is_quarantined") == True)
        .drop("is_quarantined")
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# =============================================================================
# ETIQUETAS — Tabla limpia
# =============================================================================

@dp.table(
    name="silver_labels",
    comment="Etiquetas limpias con delayed feedback validado.",
)
def silver_labels():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA}.bronze_labels")
        .withColumn("is_quarantined", _build_quarantine_flag(LABEL_RULES))
        .filter(F.col("is_quarantined") == False)
        .drop("is_quarantined")
    )


# =============================================================================
# TABLA DE HECHOS UNIFICADA — Stream-Stream Join con Watermark
# =============================================================================

@dp.table(
    name="silver_inspections_labeled",
    comment=(
        "Tabla de hechos unificada: inspecciones enriquecidas con su etiqueta de defecto. "
        "Join stream-stream con watermark de 30 días para gestionar delayed feedback."
    ),
)
def silver_inspections_labeled():
    inspections = (
        spark.readStream.table(f"{CATALOG}.{SCHEMA}.silver_inspections")
        .withWatermark("timestamp", WATERMARK_DELAY)
        .select(
            "unit_id", "timestamp", "machine_id", "line_id", "shift",
            "supplier_id", "material_batch_id",
            "temperature_celsius", "pressure_bar", "vibration_mm_s",
            "voltage_v", "current_ma", "humidity_pct", "particle_count_m3",
            "solder_thickness_um", "alignment_error_um", "optical_density",
            "tool_wear_pct", "time_since_maintenance_h", "production_speed_pct",
            "operator_experience_yrs", "cycle_time_s",
            "spc_xbar", "spc_range", "cumulative_defect_rate_shift",
        )
    )

    labels = (
        spark.readStream.table(f"{CATALOG}.{SCHEMA}.silver_labels")
        .withWatermark("label_available_date", WATERMARK_DELAY)
        .select("unit_id", "is_defective", "label_available_date")
    )

    return inspections.join(
        labels,
        on="unit_id",
        how="left",
    ).withColumn("join_timestamp", F.current_timestamp())