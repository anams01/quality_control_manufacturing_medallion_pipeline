"""
01_bronze_ingestion.py
======================
Capa Bronce — Ingesta raw de datos desde la landing_zone.
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CATALOG        = "workspace"
SCHEMA_VOLUME  = "quality_control_manufacturing"  # donde está landing_zone
VOLUME         = "landing_zone"

BASE_PATH          = f"/Volumes/{CATALOG}/{SCHEMA_VOLUME}/{VOLUME}"
CONTEXT_PATH       = f"{BASE_PATH}/context"
EVENTS_PATH        = f"{BASE_PATH}/events"
SOURCE_BUFFER_PATH = f"{BASE_PATH}/source_buffer"
CHECKPOINT_BASE    = f"{BASE_PATH}/_checkpoints"


# =============================================================================
# HELPER: columnas de auditoría
# =============================================================================

def _audit_cols(df):
    return df.withColumn(
        "ingestion_timestamp", F.current_timestamp()
    ).withColumn(
        "source_file", F.col("_metadata.file_path")
    )


# =============================================================================
# CONTEXT — Modo BATCH
# =============================================================================

@dp.table(name="bronze_machines",
          comment="Catálogo de las 32 máquinas con sus características técnicas.")
def bronze_machines():
    return _audit_cols(
        spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{CONTEXT_PATH}/machines.csv")
    )


@dp.table(name="bronze_lines",
          comment="Las 4 líneas de producción con capacidad y clase de sala limpia.")
def bronze_lines():
    return _audit_cols(
        spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{CONTEXT_PATH}/lines.csv")
    )


@dp.table(name="bronze_suppliers",
          comment="Proveedores de componentes, incluido SUP_DELTA (nuevo desde 2025-03).")
def bronze_suppliers():
    return _audit_cols(
        spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{CONTEXT_PATH}/suppliers.csv")
    )


@dp.table(name="bronze_operators",
          comment="120 operarios con nivel de experiencia y turno asignado.")
def bronze_operators():
    return _audit_cols(
        spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{CONTEXT_PATH}/operators.csv")
    )


@dp.table(name="bronze_maintenance",
          comment="Historial de mantenimientos preventivos y correctivos por máquina.")
def bronze_maintenance():
    return _audit_cols(
        spark.read.option("header", True).option("inferSchema", True)
        .csv(f"{CONTEXT_PATH}/maintenance.csv")
    )


# =============================================================================
# EVENTS — Modo STREAMING con Auto Loader
# =============================================================================

@dp.table(
    name="bronze_inspections",
    comment="Eventos de inspección: 1.000.000 unidades/mes. Histórico 2023-2024 + 2025.",
)
def bronze_inspections():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", True)
        .option("cloudFiles.schemaLocation",
                f"{CHECKPOINT_BASE}/bronze_inspections/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{EVENTS_PATH}/inspections/")
        .transform(_audit_cols)
    )


@dp.append_flow(target="bronze_inspections")
def ingest_inspections_buffer():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", True)
        .option("cloudFiles.schemaLocation",
                f"{CHECKPOINT_BASE}/bronze_inspections_buffer/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{SOURCE_BUFFER_PATH}/inspections/")
        .transform(_audit_cols)
    )


@dp.table(
    name="bronze_labels",
    comment="Etiquetas de defecto con delayed feedback. Separadas de inspections.",
)
def bronze_labels():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", True)
        .option("cloudFiles.schemaLocation",
                f"{CHECKPOINT_BASE}/bronze_labels/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{EVENTS_PATH}/labels/")
        .transform(_audit_cols)
    )


@dp.append_flow(target="bronze_labels")
def ingest_labels_buffer():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", True)
        .option("cloudFiles.schemaLocation",
                f"{CHECKPOINT_BASE}/bronze_labels_buffer/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{SOURCE_BUFFER_PATH}/labels/")
        .transform(_audit_cols)
    )