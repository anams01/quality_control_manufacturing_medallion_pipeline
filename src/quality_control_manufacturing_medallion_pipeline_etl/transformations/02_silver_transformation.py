"""
02_silver_transformation.py
============================
Capa Plata — Limpieza, validación, cuarentena y enriquecimiento.
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F
from rules import get_rules_by_tag

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"              # donde se crean las tablas
SCHEMA_VOLUME = "quality_control_manufacturing"  # donde está landing_zone
VOLUME        = "landing_zone"

BASE_PATH       = f"/Volumes/{CATALOG}/{SCHEMA_VOLUME}/{VOLUME}"
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
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_inspections")
        .withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
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
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_inspections")
        .withColumn("timestamp", F.to_timestamp(F.col("timestamp")))
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
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_labels")
        .withColumn("label_available_date", F.to_timestamp(F.col("label_available_date")))
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
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_labels")
        .withColumn("label_available_date", F.to_timestamp(F.col("label_available_date")))
        .withColumn("is_quarantined", _build_quarantine_flag(LABEL_RULES))
        .filter(F.col("is_quarantined") == False)
        .drop("is_quarantined")
    )


# =============================================================================
# TABLA DE HECHOS UNIFICADA — Stream-Stream Join con Watermark
# =============================================================================

@dp.table(
    name="silver_inspections_labeled",
    comment="Tabla de hechos unificada: inspecciones con etiqueta de defecto.",
)
def silver_inspections_labeled():
    inspections = (
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.silver_inspections")
        .withWatermark("timestamp", WATERMARK_DELAY)
        .select(
            "unit_id", "timestamp", "machine_id", "line_id", "shift",
            "supplier_id", "material_batch_id",
            "temperature_celsius", "pressure_bar", "vibration_mm_s",
            "voltage_v", "current_ma", "humidity_pct", "particle_count_m3",
            "solder_thickness_um", "alignment_error_um", "optical_density",
            "tool_wear_pct", "time_since_maintenance_h", "production_speed_pct",
            "operator_experience_yrs", "cycle_time_s",
        )
    )

    labels = (
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.silver_labels")
        .withWatermark("label_available_date", WATERMARK_DELAY)
        .select("unit_id", "is_defective", "label_available_date")
    )

    return inspections.join(
        labels,
        on=[
            inspections.unit_id == labels.unit_id,
            labels.label_available_date >= inspections.timestamp,
            labels.label_available_date <= inspections.timestamp + F.expr("INTERVAL 30 DAYS"),
        ],
        how="left",
    ).select(
        inspections.unit_id,
        inspections.timestamp,
        inspections.machine_id, inspections.line_id, inspections.shift,
        inspections.supplier_id, inspections.material_batch_id,
        inspections.temperature_celsius, inspections.pressure_bar,
        inspections.vibration_mm_s, inspections.voltage_v,
        inspections.current_ma, inspections.humidity_pct,
        inspections.particle_count_m3, inspections.solder_thickness_um,
        inspections.alignment_error_um, inspections.optical_density,
        inspections.tool_wear_pct, inspections.time_since_maintenance_h,
        inspections.production_speed_pct, inspections.operator_experience_yrs,
        inspections.cycle_time_s,
        labels.is_defective,
        labels.label_available_date,
        F.current_timestamp().alias("join_timestamp"),
    )