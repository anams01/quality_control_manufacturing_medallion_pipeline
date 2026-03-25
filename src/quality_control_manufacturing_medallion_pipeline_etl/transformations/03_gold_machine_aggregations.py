"""
03_gold_machine_aggregations.py
================================
Capa Oro — Agregaciones dinámicas de comportamiento por máquina.

En lugar de hacer JOIN entre vistas streaming (no soportado en Spark),
se crean 4 tablas independientes, una por ventana temporal.

Tablas generadas:
  - gold_machine_agg_1h   : métricas por máquina en ventana de 1 hora
  - gold_machine_agg_24h  : métricas por máquina en ventana de 24 horas
  - gold_machine_agg_7d   : métricas por máquina en ventana de 7 días
  - gold_machine_agg_30d  : métricas por máquina en ventana de 30 días
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"
WATERMARK     = "30 days"


@dp.table(
    name="gold_machine_agg_1h",
    comment="Métricas de calidad por máquina en ventana de 1 hora. Para alertas en tiempo real.",
    table_properties={"delta.enableChangeDataFeed": "true"},
)
def gold_machine_agg_1h():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.silver_inspections_labeled")
        .withWatermark("timestamp", WATERMARK)
        .groupBy("machine_id", "line_id", F.window("timestamp", "1 hour"))
        .agg(
            F.count("unit_id").alias("total_units"),
            F.sum("is_defective").alias("defects"),
            F.mean("vibration_mm_s").alias("avg_vibration"),
            F.mean("tool_wear_pct").alias("avg_tool_wear"),
            F.mean("temperature_celsius").alias("avg_temperature"),
            F.mean("solder_thickness_um").alias("avg_solder_thickness"),
            F.mean("alignment_error_um").alias("avg_alignment_error"),
        )
        .select(
            "machine_id", "line_id",
            F.col("window.end").alias("window_end"),
            F.lit("1h").alias("window_size"),
            "total_units", "defects",
            (F.col("defects") / F.col("total_units")).alias("defect_rate"),
            "avg_vibration", "avg_tool_wear", "avg_temperature",
            "avg_solder_thickness", "avg_alignment_error",
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )


@dp.table(
    name="gold_machine_agg_24h",
    comment="Métricas de calidad por máquina en ventana de 24 horas. Para tendencia diaria.",
    table_properties={"delta.enableChangeDataFeed": "true"},
)
def gold_machine_agg_24h():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.silver_inspections_labeled")
        .withWatermark("timestamp", WATERMARK)
        .groupBy("machine_id", "line_id", F.window("timestamp", "24 hours"))
        .agg(
            F.count("unit_id").alias("total_units"),
            F.sum("is_defective").alias("defects"),
            F.mean("vibration_mm_s").alias("avg_vibration"),
            F.mean("tool_wear_pct").alias("avg_tool_wear"),
            F.mean("temperature_celsius").alias("avg_temperature"),
            F.mean("solder_thickness_um").alias("avg_solder_thickness"),
            F.mean("alignment_error_um").alias("avg_alignment_error"),
        )
        .select(
            "machine_id", "line_id",
            F.col("window.end").alias("window_end"),
            F.lit("24h").alias("window_size"),
            "total_units", "defects",
            (F.col("defects") / F.col("total_units")).alias("defect_rate"),
            "avg_vibration", "avg_tool_wear", "avg_temperature",
            "avg_solder_thickness", "avg_alignment_error",
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )


@dp.table(
    name="gold_machine_agg_7d",
    comment="Métricas de calidad por máquina en ventana de 7 días. Para tendencia semanal.",
    table_properties={"delta.enableChangeDataFeed": "true"},
)
def gold_machine_agg_7d():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.silver_inspections_labeled")
        .withWatermark("timestamp", WATERMARK)
        .groupBy("machine_id", "line_id", F.window("timestamp", "7 days"))
        .agg(
            F.count("unit_id").alias("total_units"),
            F.sum("is_defective").alias("defects"),
            F.mean("vibration_mm_s").alias("avg_vibration"),
            F.mean("tool_wear_pct").alias("avg_tool_wear"),
            F.mean("solder_thickness_um").alias("avg_solder_thickness"),
        )
        .select(
            "machine_id", "line_id",
            F.col("window.end").alias("window_end"),
            F.lit("7d").alias("window_size"),
            "total_units", "defects",
            (F.col("defects") / F.col("total_units")).alias("defect_rate"),
            "avg_vibration", "avg_tool_wear", "avg_solder_thickness",
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )


@dp.table(
    name="gold_machine_agg_30d",
    comment="Métricas de calidad por máquina en ventana de 30 días. Baseline mensual.",
    table_properties={"delta.enableChangeDataFeed": "true"},
)
def gold_machine_agg_30d():
    return (
        spark.readStream.table(f"{CATALOG}.{SCHEMA_TABLES}.silver_inspections_labeled")
        .withWatermark("timestamp", WATERMARK)
        .groupBy("machine_id", "line_id", F.window("timestamp", "30 days"))
        .agg(
            F.count("unit_id").alias("total_units"),
            F.sum("is_defective").alias("defects"),
            F.mean("vibration_mm_s").alias("avg_vibration"),
            F.mean("tool_wear_pct").alias("avg_tool_wear"),
            F.mean("solder_thickness_um").alias("avg_solder_thickness"),
        )
        .select(
            "machine_id", "line_id",
            F.col("window.end").alias("window_end"),
            F.lit("30d").alias("window_size"),
            "total_units", "defects",
            (F.col("defects") / F.col("total_units")).alias("defect_rate"),
            "avg_vibration", "avg_tool_wear", "avg_solder_thickness",
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )