"""
03_gold_machine_profile.py
===========================
Capa Oro — Perfiles estáticos de máquinas y proveedores.

Deriva características estáticas o de actualización lenta:
  - Perfil de máquina: tipo, línea, antigüedad, baseline de vibración
  - Perfil de proveedor: calidad histórica, grosor de soldadura esperado

Tabla generada:
  - gold_machine_profile : perfil enriquecido por machine_id
"""

import pyspark.pipelines as dp
from pyspark.sql import functions as F

CATALOG       = "workspace"
SCHEMA_TABLES = "ana_martin17"


@dp.table(
    name="gold_machine_profile",
    comment=(
        "Perfil estático por máquina enriquecido con datos de contexto. "
        "Incluye antigüedad, tipo de máquina, línea asignada y proveedor predominante. "
        "Diseñado para ser consumido por el feature store en entrenamiento e inferencia."
    ),
    table_properties={"delta.enableChangeDataFeed": "true"},
    schema="""
        machine_id              STRING      NOT NULL,
        line_id                 STRING,
        machine_type            STRING,
        installation_date       DATE,
        machine_age_days        INT,
        nominal_cycle_time_s    DOUBLE,
        vibration_baseline_mm_s DOUBLE,
        wear_rate_pct_month     DOUBLE,
        clean_room_class        STRING,
        line_capacity_units_day INT,
        ingestion_timestamp     TIMESTAMP,
        CONSTRAINT machine_pk PRIMARY KEY (machine_id)
    """,
)
def gold_machine_profile():
    machines = spark.read.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_machines")
    lines    = spark.read.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_lines")

    return (
        machines
        .join(lines.select("line_id", "clean_room_class", "capacity_units_day"),
              on="line_id", how="left")
        .withColumn(
            "machine_age_days",
            F.datediff(F.current_date(), F.to_date(F.col("installation_date")))
        )
        .select(
            "machine_id",
            "line_id",
            "machine_type",
            F.to_date("installation_date").alias("installation_date"),
            "machine_age_days",
            "nominal_cycle_time_s",
            "vibration_baseline_mm_s",
            "wear_rate_pct_month",
            "clean_room_class",
            F.col("capacity_units_day").alias("line_capacity_units_day"),
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )


@dp.table(
    name="gold_supplier_profile",
    comment=(
        "Perfil estático por proveedor con métricas de calidad esperadas. "
        "Permite al modelo detectar cambios de proveedor como señal de riesgo."
    ),
    table_properties={"delta.enableChangeDataFeed": "true"},
    schema="""
        supplier_id                 STRING  NOT NULL,
        supplier_name               STRING,
        country                     STRING,
        onboarding_date             DATE,
        solder_thickness_mean_um    DOUBLE,
        quality_rating              DOUBLE,
        is_new_supplier             BOOLEAN,
        ingestion_timestamp         TIMESTAMP,
        CONSTRAINT supplier_pk PRIMARY KEY (supplier_id)
    """,
)
def gold_supplier_profile():
    return (
        spark.read.table(f"{CATALOG}.{SCHEMA_TABLES}.bronze_suppliers")
        .withColumn(
            "is_new_supplier",
            F.col("onboarding_date") >= F.lit("2025-01-01")
        )
        .withColumn("solder_thickness_mean_um", F.col("solder_thickness_mean_um").cast("double"))
        .select(
            "supplier_id",
            "supplier_name",
            "country",
            F.to_date("onboarding_date").alias("onboarding_date"),
            "solder_thickness_mean_um",
            "quality_rating",
            "is_new_supplier",
            F.current_timestamp().alias("ingestion_timestamp"),
        )
    )
