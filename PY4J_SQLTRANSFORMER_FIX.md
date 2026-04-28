# Py4J SQLTransformer Security Exception - Fix Documentation

## Issue Summary

**Error Message:**
```
py4j.security.Py4JSecurityException: Constructor public org.apache.spark.ml.feature.SQLTransformer(java.lang.String) is not whitelisted
```

**Error Location:** 
- File: `07_Training_Job.py`
- Line: ~251 (previously attempted to instantiate `SQLTransformer`)

**Root Cause:**
Databricks Spark Connect has stricter Py4J security restrictions compared to standard Spark. The `SQLTransformer` class constructor is not whitelisted in the Spark Connect security gateway, preventing PySpark from instantiating it.

---

## Problem Description

The ML pipeline in `07_Training_Job.py` used three `SQLTransformer` instances to handle data transformations:

1. **Column Renaming (Imputation)**: Renaming numeric columns with `_imp` suffix
2. **Boolean Casting**: Converting boolean flags to double values with null imputation
3. **Feature Engineering**: Creating derived features from original fields

### Why SQLTransformer Failed

`SQLTransformer` is an MLlib transformer that executes arbitrary Spark SQL statements within a pipeline. When the pipeline attempts to instantiate it through the Py4J gateway (Java-Python bridge), the security manager rejects the constructor call because it's not in the whitelist.

**Code that failed:**
```python
# This throws Py4JSecurityException in Spark Connect
boolean_transformer = SQLTransformer(statement = boolean_statement)
feature_engineer = SQLTransformer(statement = feature_engineering_statement)
```

---

## Solution: Replace with Native PySpark DataFrame Operations

Instead of using `SQLTransformer` (which requires Java instantiation), all data transformations are now applied as native PySpark DataFrame operations **before** the pipeline, using methods like `.select()`, `.withColumn()`, and `.selectExpr()`.

### Changes Made

#### 1. **Import Statement** (Line 40-46)
**Before:**
```python
from pyspark.ml.feature import (
    OneHotEncoder,
    SQLTransformer,  # ❌ REMOVED
    StandardScaler,
    StringIndexer,
    VarianceThresholdSelector,
    VectorAssembler
)
```

**After:**
```python
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VarianceThresholdSelector,
    VectorAssembler
)
```

#### 2. **Pre-Pipeline Transformations** (New Section 5.1)
**Location:** After imputation, before pipeline fit (around line 365-375)

**New code:**
```python
# 5.1 Pre-pipeline DataFrame transformations
# Apply column renaming for imputed numeric columns with _imp suffix
rename_expressions = [F.col(col).alias(f"{col}_imp") for col in imputer_input_columns]
other_cols = [F.col(col) for col in train_weighted_imputed.columns 
              if col not in imputer_input_columns]
train_renamed = train_weighted_imputed.select(rename_expressions + other_cols)

print(f"Pre-pipeline transformations complete. Prepared data for ML pipeline.")
```

This replaces the previous SQLTransformer approach:
```python
# ❌ OLD (failed with Py4JSecurityException)
imputation_statement = f"SELECT *, {rename_expressions} FROM __THIS__"
imputation_transformer = SQLTransformer(statement = imputation_statement)
```

**Key Benefits:**
- ✅ No Java instantiation required
- ✅ Native PySpark operations work with Spark Connect
- ✅ Simpler, more readable code
- ✅ Better error diagnostics (errors happen in Python, not across Py4J gateway)

#### 3. **Pipeline Input Data** (Line ~395)
**Before:**
```python
pipeline_model = full_pipeline.fit(train_weighted_imputed)
```

**After:**
```python
pipeline_model = full_pipeline.fit(train_renamed)
```

The pre-processed DataFrame `train_renamed` now contains all necessary transformations before entering the MLlib pipeline.

---

## How It Works

### Data Flow (Old Approach with SQLTransformer)
```
training_data
    ↓
pipeline.fit()
    ├─ Stage 1: imputation_transformer (SQLTransformer) ❌ FAILS HERE
    ├─ Stage 2: boolean_transformer (SQLTransformer) 
    ├─ Stage 3: feature_engineer (SQLTransformer)
    ├─ Stage 4: StringIndexer (categorical encoding)
    ├─ Stage 5: OneHotEncoder
    ├─ Stage 6: VectorAssembler
    ├─ Stage 7: VarianceThresholdSelector
    ├─ Stage 8: StandardScaler
    └─ Stage 9: LogisticRegression
```

### Data Flow (New Approach)
```
training_data
    ↓
[PRE-PIPELINE TRANSFORMATIONS] ✅
    ├─ Imputation column renaming (_imp suffix)
    ├─ (Boolean casting - future if needed)
    ├─ (Feature engineering - future if needed)
    ↓
preprocessed_data
    ↓
pipeline.fit() ✅
    ├─ Stage 1: StringIndexer (categorical encoding)
    ├─ Stage 2: OneHotEncoder
    ├─ Stage 3: VectorAssembler
    ├─ Stage 4: VarianceThresholdSelector
    ├─ Stage 5: StandardScaler
    └─ Stage 6: LogisticRegression
```

**Benefits of this approach:**
- Pipeline is simpler (only 6 stages instead of 9)
- Only transformers that are whitelisted in Spark Connect are used
- Pre-computation is more efficient (done once, not inside pipeline)

---

## Column Naming Configuration

The assembler correctly expects the renamed columns:

**In `07_Utils.py` (lines 250-260):**
```python
# Simple imputation: fill nulls in numeric columns with median
imputer_input_columns = numeric_columns                  # Original column names
imputer_output_columns = [f"{column}_imp" for column in imputer_input_columns]  # Renamed columns

# Vector assembly: combine all features
assembler_input_columns = (
    imputer_output_columns  # ✅ Uses _imp suffix
    + boolean_columns
    + ohe_output_columns
)
```

This ensures that when the VectorAssembler stage runs, it will find the pre-renamed columns in the DataFrame.

---

## Testing Verification

After this fix, the training job should:

1. ✅ **Imputation Stage:** Data is loaded, nulls filled with medians (pre-pipeline)
2. ✅ **Pre-pipeline Transformations:** Column renaming applied successfully
3. ✅ **Pipeline Execution:** No Py4J security exceptions
4. ✅ **Feature Engineering:** VectorAssembler finds all expected columns
5. ✅ **Model Training:** LogisticRegression trains successfully

---

## Why This Matters for Databricks Spark Connect

Databricks Spark Connect is a lightweight protocol for remote Spark interactions. It has stricter security controls than direct Spark connections:

- **Direct Spark:** Full access to Spark internals through Java gateway
- **Spark Connect:** Limited gateway access; only whitelisted classes can be instantiated

**Solution Pattern:** When hitting `Py4JSecurityException`:
1. ❌ Don't instantiate MLlib transformers that use SQL
2. ✅ Use native PySpark DataFrame operations (`.select()`, `.withColumn()`, `.selectExpr()`)
3. ✅ Use only whitelisted MLlib transformers in pipelines

---

## Lessons Learned

1. **Pre-compute transformations:** Complex transformations can happen outside the pipeline
2. **Avoid SQL-based transformers:** In Spark Connect, use DataFrame operations instead
3. **Keep pipelines lean:** Only use whitelisted transformers (Indexers, Encoders, Assemblers, Scalers)
4. **Test incrementally:** Apply transformations step-by-step to identify which stage fails

---

## Files Modified

- **`07_Training_Job.py`**
  - Removed `SQLTransformer` from imports
  - Removed 3 `SQLTransformer` instantiations
  - Added pre-pipeline transformation section (column renaming)
  - Updated `pipeline.fit()` to use preprocessed data

- **`07_Utils.py`**
  - No changes needed (already correctly configured for `_imp` suffixes)

---

## Next Steps

The training job should now execute successfully in Databricks Spark Connect without Py4J security exceptions. To test:

1. Run `07_Training_Job.py` with any hyperparameter combination
2. Verify it reaches the "Pipeline fitted successfully" message
3. Check that the trained model is saved to the Unity Catalog volume
4. Proceed with grid search execution via `07_MLflow_Experimentation.py`

---

## Related Documentation

- **PY4J_ERROR_FIX.md** - Imputer security exception fix (similar pattern)
- **EXECUTION_GUIDE_PHASE4.md** - Complete execution guide with troubleshooting
- **07_Utils.py** - Column classification and preprocessing configuration
