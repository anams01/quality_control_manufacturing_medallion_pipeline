"""
Shared utilities for manufacturing quality control and defect detection experimentation.
Adapted from credit card fraud detection baseline.
"""

###############################################################################
# Imports
###############################################################################

import os
from datetime import datetime, timedelta
from pathlib import Path

from dateutil.relativedelta import relativedelta

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
matplotlib.use("Agg")  # Non-interactive backend: safe on cluster drivers with no display

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VarianceThresholdSelectorModel
from pyspark.sql import functions as F

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)


###############################################################################
# Data and infrastructure
###############################################################################

CATALOG = "workspace"
DATABASE = "ana_martin17"
TRAINING_TABLE = f"{CATALOG}.{DATABASE}.gold_inspection_training_dataset"

catalog = CATALOG
database = DATABASE
training_table = TRAINING_TABLE

uc_volume_path = Path("/") / "Volumes" / CATALOG / DATABASE / "ml_artifacts"

# Required for logging models on serverless clusters
os.environ["MLFLOW_DFS_TMP"] = str(uc_volume_path)


###############################################################################
# Column configuration
###############################################################################

LABEL_COLUMN = "is_defective"
CLASS_WEIGHT_COLUMN = "class_weight"
FEATURES_COLUMN = "features_scaled"
DATE_COLUMN = "timestamp"
UNIT_ID_COLUMN = "unit_id"

label_column = LABEL_COLUMN
class_weight_column = CLASS_WEIGHT_COLUMN
features_column = FEATURES_COLUMN
date_column = DATE_COLUMN
unit_id_column = UNIT_ID_COLUMN


###############################################################################
# Temporal split configuration
###############################################################################

TRAINING_WINDOW_MONTHS = 24
VALIDATION_WINDOW_MONTHS = 6

seed = 45127


###############################################################################
# Project metadata
###############################################################################

current_user = spark.sql("SELECT current_user()").collect()[0][0]
project = "manufacturing_quality_control"
team = "ml_engineering"
task = "binary_classification"
environment = "development"
algorithm_family = "logistic_regression"
framework = "pyspark.ml"

uc_model_name = f"{catalog}.{database}.defect_detection_lr_pipeline"

mlflow_experiment_name = "defect_detection_training"
mlflow_experiment_path = str(
    Path("/") / "Workspace" / "Users" / current_user / ".experiments" / database / mlflow_experiment_name
)


###############################################################################
# Raw data load
###############################################################################

df_raw = spark.table(training_table)

print(f"Total rows: {df_raw.count():,}")
print(f"Total columns: {len(df_raw.columns)}")
print()


###############################################################################
# Temporal split and inverse-frequency class weights
###############################################################################

properties_df = spark.sql(f"SHOW TBLPROPERTIES {training_table}")
semantic_version_row = properties_df.filter("key = 'ml.delta_semantic_version'").first()
delta_semantic_version = int(semantic_version_row["value"]) if semantic_version_row else 0

if delta_semantic_version == 0:
    # First version: 70% train, 15% validation, 15% test
    all_dates = df_raw.agg(F.min(F.col(date_column)), F.max(F.col(date_column))).collect()[0]
    min_date = all_dates[0]
    max_date = all_dates[1]
    date_range = (max_date - min_date).days
    
    train_days = int(0.70 * date_range)
    validation_days = int(0.15 * date_range)
    
    train_end = min_date + timedelta(days=train_days)
    validation_end = train_end + timedelta(days=validation_days)
    test_end = max_date
else:
    # Subsequent versions: use rolling window
    previous_max_date_row = properties_df.filter("key = 'ml.data_previous_max_date'").first()
    validation_end = datetime.strptime(previous_max_date_row["value"], "%Y-%m-%d")
    train_end = validation_end - relativedelta(months=VALIDATION_WINDOW_MONTHS)
    train_start = train_end - relativedelta(months=TRAINING_WINDOW_MONTHS)

train_start_date = train_end - relativedelta(months=TRAINING_WINDOW_MONTHS)
train_end_date = train_end
validation_start_date = train_end + timedelta(days=1)
validation_end_date = validation_end
test_start_date = validation_end + timedelta(days=1)
test_end_date = df_raw.agg(F.max(F.col(date_column)).alias("max_date")).collect()[0]["max_date"]

train_df = df_raw.filter(
    (F.col(date_column) >= train_start_date) & (F.col(date_column) <= train_end_date)
)
validation_df = df_raw.filter(
    (F.col(date_column) >= validation_start_date) & (F.col(date_column) <= validation_end_date)
)
test_df = df_raw.filter(
    (F.col(date_column) >= test_start_date) & (F.col(date_column) <= test_end_date)
)

print(f"Semantic version: {delta_semantic_version}")
print()
print(f"Train period: {train_start_date.strftime('%Y-%m-%d')} → {train_end_date.strftime('%Y-%m-%d')}")
print(f"Validation period: {validation_start_date.strftime('%Y-%m-%d')} → {validation_end_date.strftime('%Y-%m-%d')}")
print(f"Test period: {test_start_date.strftime('%Y-%m-%d')} → {test_end_date.strftime('%Y-%m-%d')}")
print()
print(f"Train rows: {train_df.count():,}")
print(f"Validation rows: {validation_df.count():,}")
print(f"Test rows: {test_df.count():,}")
print()


def apply_class_weights(df):
    """
    Calculates and applies inverse-frequency class weights dynamically
    based on the exact distribution of the provided `DataFrame`.
    """
    n_total = df.count()
    n_defective = df.filter(F.col(label_column) == 1).count()
    n_good = n_total - n_defective

    pct_defective = 100 * n_defective / n_total if n_total > 0 else 0.0
    pct_good = 100 * n_good / n_total if n_total > 0 else 0.0

    weight_defective = n_total / (2.0 * n_defective) if n_defective > 0 else 1.0
    weight_good = n_total / (2.0 * n_good) if n_good > 0 else 1.0

    print(f"Total rows for training: {n_total:,}")
    print(f"Defective: {n_defective:,} ({pct_defective:.2f}%), weight = {weight_defective:.2f}")
    print(f"Good: {n_good:,} ({pct_good:.2f}%), weight = {weight_good:.2f}")
    print()

    weighted_df = df.withColumn(
        class_weight_column,
        F.when(F.col(label_column) == 1.0, weight_defective).otherwise(weight_good)
    )

    return weighted_df


###############################################################################
# Column classification
###############################################################################

# Type sets used to route each field to the correct pipeline stage
numeric_types = {"IntegerType", "LongType", "FloatType", "DoubleType", "DecimalType"}
boolean_types = {"BooleanType"}
categorical_types = {"StringType"}

# Columns to exclude from feature vector
exclude_columns = [
    label_column,
    unit_id_column,
    "machine_id",
    "line_id",
    "supplier_id",
    "material_batch_id",
    date_column,
    "ingestion_timestamp"
]

# Identify column types
numeric_columns = []
boolean_columns = []
categorical_columns = []

for field in df_raw.schema.fields:
    column_name = field.name
    type_name = type(field.dataType).__name__
    if column_name in exclude_columns:
        continue
    if type_name in numeric_types:
        numeric_columns.append(column_name)
    elif type_name in boolean_types:
        boolean_columns.append(column_name)
    elif type_name in categorical_types:
        categorical_columns.append(column_name)

print(f"Numeric ({len(numeric_columns)}): {numeric_columns}")
print(f"Boolean ({len(boolean_columns)}): {boolean_columns}")
print(f"Categorical ({len(categorical_columns)}): {categorical_columns}")
print()


###############################################################################
# Preprocessing configuration
###############################################################################

# Simple imputation: fill nulls in numeric columns with median
imputer_input_columns = numeric_columns
imputer_output_columns = [f"{column}_imp" for column in imputer_input_columns]

# Categorical encoding
string_indexer_input_columns = categorical_columns
string_indexer_output_columns = [f"{column}_idx" for column in categorical_columns]

ohe_input_columns = string_indexer_output_columns
ohe_output_columns = [f"{column}_ohe" for column in categorical_columns]

# Vector assembly: combine all features
assembler_input_columns = (
    imputer_output_columns
    + boolean_columns
    + ohe_output_columns
)

print(f"Assembler input ({len(assembler_input_columns)}): {assembler_input_columns}")
print()
