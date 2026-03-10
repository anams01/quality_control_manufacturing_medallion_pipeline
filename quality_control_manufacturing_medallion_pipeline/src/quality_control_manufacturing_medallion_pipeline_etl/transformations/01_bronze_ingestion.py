"""
01_bronze_ingestion.py
======================
Capa Bronce — Ingesta raw de datos desde la landing_zone.

Convierte los archivos originales (.csv, .json) al formato Delta Lake,
añadiendo metadatos de auditoría (ingestion_timestamp, source_file).

Tablas generadas:
  - bronze_machines        : catálogo de máquinas (batch)
  - bronze_lines           : líneas de producción (batch)
  - bronze_suppliers       : proveedores (batch)
  - bronze_operators       : operarios (batch)
  - bronze_maintenance     : historial de mantenimiento (batch)
  - bronze_inspections     : eventos de inspección por unidad (streaming)
  - bronze_labels          : etiquetas de defecto con delayed feedback (streaming)
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

CATALOG    = "workspace"
SCHEMA     = "quality_control_manufacturing"   # tu schema en Databricks
VOLUME     = "landing_zone"

BASE_PATH  = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

CONTEXT_PATH      = f"{BASE_PATH}/context"
EVENTS_PATH       = f"{BASE_PATH}/events"
SOURCE_BUFFER_PATH = f"{BASE_PATH}/source_buffer"

CHECKPOINT_BASE   = f"{BASE_PATH}/_checkpoints"


# =============================================================================
# HELPER: columnas de auditoría
# =============================================================================

def _audit_cols(df):
    """Añade columnas técnicas de trazabilidad."""
    return df.withColumn(
        "ingestion_timestamp", F.current_timestamp()
    ).withColumn(
        "source_file", F.input_file_name()
    )


# =============================================================================
# TABLAS CONTEXT — Modo BATCH (datos maestros estáticos)
# =============================================================================

@dp.table(
    name="bronze_machines",
    comment="Catálogo de las 32 máquinas de producción con sus características técnicas.",
)
def bronze_machines():
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{CONTEXT_PATH}/machines.csv"
    )
    return _audit_cols(df)


@dp.table(
    name="bronze_lines",
    comment="Las 4 líneas de producción con su capacidad y clase de sala limpia.",
)
def bronze_lines():
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{CONTEXT_PATH}/lines.csv"
    )
    return _audit_cols(df)


@dp.table(
    name="bronze_suppliers",
    comment="Proveedores de componentes, incluyendo SUP_DELTA (nuevo desde 2025-03).",
)
def bronze_suppliers():
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{CONTEXT_PATH}/suppliers.csv"
    )
    return _audit_cols(df)


@dp.table(
    name="bronze_operators",
    comment="Roster de 120 operarios con nivel de experiencia y turno asignado.",
)
def bronze_operators():
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{CONTEXT_PATH}/operators.csv"
    )
    return _audit_cols(df)


@dp.table(
    name="bronze_maintenance",
    comment="Historial de mantenimientos preventivos y correctivos por máquina.",
)
def bronze_maintenance():
    df = spark.read.option("header", True).option("inferSchema", True).csv(
        f"{CONTEXT_PATH}/maintenance.csv"
    )
    return _audit_cols(df)


# =============================================================================
# TABLAS EVENTS — Modo STREAMING con Auto Loader (eventos incrementales)
# =============================================================================

@dp.create_streaming_table(
    name="bronze_inspections",
    comment=(
        "Eventos de inspección unitaria: lecturas de sensores y parámetros de proceso. "
        "1.000.000 unidades/mes. Periodo histórico: 2023-01 a 2024-12. "
        "Datos de producción (con drift): 2025-01 a 2025-06 en source_buffer."
    ),
)
def bronze_inspections_historical():
    pass


@dp.append_flow(target="bronze_inspections")
def ingest_inspections_historical():
    """Ingesta histórica 2023-2024 desde events/."""
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", True)
        .option("cloudFiles.schemaLocation",
                f"{CHECKPOINT_BASE}/bronze_inspections_historical/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{EVENTS_PATH}/inspections/")
        .transform(_audit_cols)
    )


@dp.append_flow(target="bronze_inspections")
def ingest_inspections_buffer():
    """Ingesta datos de producción 2025 desde source_buffer/."""
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


@dp.create_streaming_table(
    name="bronze_labels",
    comment=(
        "Etiquetas de defecto con delayed feedback: is_defective + label_available_date. "
        "Separadas físicamente de inspections para simular feedback tardío real "
        "(confirmación del defecto puede tardar hasta 30 días)."
    ),
)
def bronze_labels_table():
    pass


@dp.append_flow(target="bronze_labels")
def ingest_labels_historical():
    """Etiquetas históricas 2023-2024."""
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", True)
        .option("cloudFiles.schemaLocation",
                f"{CHECKPOINT_BASE}/bronze_labels_historical/schema")
        .option("rescuedDataColumn", "_rescued_data")
        .load(f"{EVENTS_PATH}/labels/")
        .transform(_audit_cols)
    )


@dp.append_flow(target="bronze_labels")
def ingest_labels_buffer():
    """Etiquetas de producción 2025."""
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