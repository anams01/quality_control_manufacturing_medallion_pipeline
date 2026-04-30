"""
03_gold_machine_aggregations.py
================================
Capa Oro — Agregaciones dinámicas de comportamiento por máquina.

Se crean 4 tablas independientes, una por ventana temporal, para cumplir con
las restricciones de Spark Structured Streaming en DLT:

  - gold_machine_agg_1h   : métricas por máquina en ventana de 1 hora
  - gold_machine_agg_24h  : métricas por máquina en ventana de 24 horas
  - gold_machine_agg_7d   : métricas por máquina en ventana de 7 días
  - gold_machine_agg_30d  : métricas por máquina en ventana de 30 días

CORRECCIONES respecto a la versión original:
  1. Se elimina el uso de pyspark.sql.Window sobre un DataFrame de readStream.
     Las Window functions analíticas (rangeBetween) no están soportadas sobre
     streams en Spark Structured Streaming y provocan AnalysisException.
  2. Se sustituye por groupBy + F.window(), que es el patrón correcto para
     agregaciones con ventana temporal en streaming DLT.
  3. Se separa en 4 tablas independientes (una por ventana), como indica el
     docstring original. El enfoque de una única tabla con 4 ventanas anidadas
     en un solo stream no es viable con la API de groupBy+window.
  4. Se añade withWatermark() obligatorio para que DLT gestione el estado y
     el late data de cada ventana.
  5. Se unifica la API: solo se usa dlt (se elimina el import de Window).
"""

import dlt
from pyspark.sql import functions as F

WATERMARK_DELAY = "2 hours"   # Tolerancia a datos tardíos; ajustar según SLA del pipeline

# Columnas de sensores a agregar en cada ventana
SENSOR_COLUMNS = [
    "temperature_celsius", "pressure_bar", "vibration_mm_s", "voltage_v", "current_ma",
    "humidity_pct", "particle_count_m3", "solder_thickness_um", "alignment_error_um",
    "optical_density", "tool_wear_pct", "time_since_maintenance_h", "production_speed_pct",
    "operator_experience_yrs", "cycle_time_s",
]


def _build_agg_exprs():
    """Genera la lista de expresiones de agregación para todos los sensores."""
    exprs = [F.avg(c).alias(f"avg_{c}") for c in SENSOR_COLUMNS]
    exprs += [
        (F.sum(F.col("is_defective").cast("double")) / F.count("is_defective") * 100)
        .alias("defect_rate_pct"),
        F.count("*").alias("inspection_count"),
    ]
    return exprs


def _make_agg_table(window_duration: str, slide_duration: str = None):
    """
    Construye el DataFrame de agregación con ventana temporal usando streaming DLT.

    Args:
        window_duration: Duración de la ventana (p.ej. "1 hour", "24 hours").
        slide_duration:  Si se especifica, usa ventanas deslizantes; si no,
                         ventanas tumbling (sin solapamiento).
    """
    stream = (
        dlt.read_stream("silver_inspections")
        .withWatermark("timestamp", WATERMARK_DELAY)
    )

    window_col = (
        F.window("timestamp", window_duration, slide_duration)
        if slide_duration
        else F.window("timestamp", window_duration)
    )

    return (
        stream
        .groupBy(window_col, F.col("machine_id"))
        .agg(*_build_agg_exprs())
        .select(
            F.col("machine_id"),
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            *[F.col(f"avg_{c}").alias(f"avg_{c}") for c in SENSOR_COLUMNS],
            F.col("defect_rate_pct"),
            F.col("inspection_count"),
        )
    )


# =============================================================================
# TABLA 1 h
# =============================================================================

@dlt.table(
    name="gold_machine_agg_1h",
    comment="Machine sensor averages and defect rate — tumbling window of 1 hour.",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true",
    },
)
def machine_agg_1h():
    return _make_agg_table("1 hour")


# =============================================================================
# TABLA 24 h
# =============================================================================

@dlt.table(
    name="gold_machine_agg_24h",
    comment="Machine sensor averages and defect rate — tumbling window of 24 hours.",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true",
    },
)
def machine_agg_24h():
    return _make_agg_table("24 hours")


# =============================================================================
# TABLA 7 d
# =============================================================================

@dlt.table(
    name="gold_machine_agg_7d",
    comment="Machine sensor averages and defect rate — tumbling window of 7 days.",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true",
    },
)
def machine_agg_7d():
    return _make_agg_table("7 days")


# =============================================================================
# TABLA 30 d
# =============================================================================

@dlt.table(
    name="gold_machine_agg_30d",
    comment="Machine sensor averages and defect rate — tumbling window of 30 days.",
    table_properties={
        "quality": "gold",
        "delta.enableChangeDataFeed": "true",
    },
)
def machine_agg_30d():
    return _make_agg_table("30 days")
