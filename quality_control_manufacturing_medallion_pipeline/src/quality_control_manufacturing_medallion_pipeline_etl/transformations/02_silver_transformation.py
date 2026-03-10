"""
02_silver_transformation.py
============================
Capa Plata — Limpieza, validación, cuarentena y enriquecimiento.

Tablas generadas:
  - silver_inspections_quarantine  : registros que no superan las reglas (DLQ)
  - silver_labels_quarantine       : etiquetas anómalas (DLQ)
  - silver_inspections             : inspecciones limpias con SCD tipo 2
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

# Watermark: máximo retraso observable entre timestamp de inspección
# y label_available_date (hasta 30 días según diseño del dataset)
WATERMARK_DELAY = "30 days"


# =============================================================================
# HELPERS
# =============================================================================

def _build_quarantine_flag(rules: dict) -> F.Column:
    """
    Construye el flag is_quarantined como la negación de todas las reglas.
    Un registro está en cuarentena si incumple AL MENOS una regla.
    """
    all_valid = F.lit(True)
    for rule in rules.values():
        all_valid = all_valid & F.expr(rule["constraint"])
    return (~all_valid).alias("is_quarantined")


# =============================================================================
# INSPECCIONES — Cuarentena + Vista limpia
# =============================================================================

INSPECTION_RULES = get_rules_by_tag("inspections")

# 1. Tabla de cuarentena (DLQ)
@dp.create_streaming_table(
    name="silver_inspections_quarantine",
    comment="Registros de inspección que no superan las reglas de calidad (Dead Letter Queue).",
)
def silver_inspections_quarantine():
    return (
        dp.read_stream("bronze_inspections")
        .withColumn("is_quarantined", _build_quarantine_flag(INSPECTION_RULES))
        .filter(F.col("is_quarantined") == True)
        .drop("is_quarantined")
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# 2. Vista limpia (solo registros válidos)
@dp.view(name="silver_inspections_clean")
def silver_inspections_clean():
    return (
        dp.read_stream("bronze_inspections")
        .withColumn("is_quarantined", _build_quarantine_flag(INSPECTION_RULES))
        .filter(F.col("is_quarantined") == False)
        .drop("is_quarantined")
    )


# 3. Tabla plata final con SCD Tipo 2 via AUTO CDC
dp.create_streaming_table(
    name="silver_inspections",
    comment="Inspecciones limpias y validadas. Histórico completo con SCD Tipo 2.",
)

dp.create_auto_cdc_flow(
    name="silver_inspections",
    source="silver_inspections_clean",
    keys=["unit_id"],
    sequence_by="timestamp",
    stored_as_scd_type=2,
    except_column_list=["ingestion_timestamp", "source_file"],
)


# =============================================================================
# ETIQUETAS — Cuarentena + Vista limpia
# =============================================================================

LABEL_RULES = get_rules_by_tag("labels")

# 1. Cuarentena de etiquetas
@dp.create_streaming_table(
    name="silver_labels_quarantine",
    comment="Etiquetas de defecto que no superan las reglas de calidad (Dead Letter Queue).",
)
def silver_labels_quarantine():
    return (
        dp.read_stream("bronze_labels")
        .withColumn("is_quarantined", _build_quarantine_flag(LABEL_RULES))
        .filter(F.col("is_quarantined") == True)
        .drop("is_quarantined")
        .withColumn("quarantine_timestamp", F.current_timestamp())
    )


# 2. Vista limpia de etiquetas
@dp.view(name="silver_labels_clean")
def silver_labels_clean():
    return (
        dp.read_stream("bronze_labels")
        .withColumn("is_quarantined", _build_quarantine_flag(LABEL_RULES))
        .filter(F.col("is_quarantined") == False)
        .drop("is_quarantined")
    )


# 3. Tabla plata de etiquetas
dp.create_streaming_table(
    name="silver_labels",
    comment="Etiquetas limpias con delayed feedback validado.",
)

dp.create_auto_cdc_flow(
    name="silver_labels",
    source="silver_labels_clean",
    keys=["unit_id"],
    sequence_by="label_available_date",
    stored_as_scd_type=1,
    except_column_list=["ingestion_timestamp", "source_file"],
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
    # Inspecciones con watermark
    inspections = (
        dp.read_stream("silver_inspections_clean")
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

    # Etiquetas con watermark
    labels = (
        dp.read_stream("silver_labels_clean")
        .withWatermark("label_available_date", WATERMARK_DELAY)
        .select("unit_id", "is_defective", "label_available_date")
    )

    # LEFT JOIN stream-stream
    return inspections.join(
        labels,
        on="unit_id",
        how="left",
    ).withColumn("join_timestamp", F.current_timestamp())