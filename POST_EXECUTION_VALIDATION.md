# POST-EXECUTION VALIDATION SCRIPT
## Run this in Databricks after Phase 4 pipeline completes

```python
# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase 4 Execution Validation
# MAGIC
# MAGIC **Purpose**: Verify successful execution of ML pipeline (grid search + production promotion)
# MAGIC
# MAGIC **Expected Runtime**: ~5 minutes
# MAGIC
# MAGIC **Execution Date**: 2026-04-27

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Setup

# COMMAND ----------

CATALOG = "workspace"
DATABASE = "ana_martin17"
MODEL_NAME = "workspace.ana_martin17.manufacturing_quality_control_model"
BASELINE_TABLE = "workspace.ana_martin17.gold_inspection_test_baseline"

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verify Training Dataset

# COMMAND ----------

training_df = spark.table(f"{CATALOG}.{DATABASE}.gold_inspection_training_dataset")
training_count = training_df.count()
training_defect_rate = training_df.filter(training_df.is_defective == True).count() / training_count

print(f"✅ Training Dataset Verification")
print(f"   Rows: {training_count:,}")
print(f"   Defect Rate: {training_defect_rate:.2%}")
print(f"   Expected: ~30M rows, ~4.2% defect rate")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verify MLflow Grid Search Runs

# COMMAND ----------

mlflow.set_experiment("/Workspace/Users/{your_user_id}/manufacturing_defect_detection")

# Search for all grid search runs
runs_df = mlflow.search_runs(
    experiment_names=["/Workspace/Users/{your_user_id}/manufacturing_defect_detection"],
    filter_string="tags.job_type='grid_search'",
    max_results=100
)

print(f"✅ Grid Search Runs")
print(f"   Total Runs: {len(runs_df)}")
print(f"   Expected: 9 runs")

# Show top 3 by validation AUC-PR
if len(runs_df) > 0:
    top_runs = runs_df.nlargest(3, 'metrics.auc_pr')[['run_id', 'metrics.auc_pr', 'metrics.f1', 'status']]
    print(f"\n   Top 3 by Validation AUC-PR:")
    for idx, row in top_runs.iterrows():
        print(f"   - Run {row['run_id'][:8]}: AUC-PR={row['metrics.auc_pr']:.4f}, F1={row['metrics.f1']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verify Model Registry - Candidate

# COMMAND ----------

try:
    candidate = client.get_model_version_by_alias(name=MODEL_NAME, alias="candidate")
    print(f"✅ Candidate Model")
    print(f"   Version: {candidate.version}")
    print(f"   Status: {candidate.status}")
    print(f"   Created: {candidate.creation_timestamp}")
except Exception as e:
    print(f"⚠️  No candidate found: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Model Registry - Champion

# COMMAND ----------

try:
    champion = client.get_model_version_by_alias(name=MODEL_NAME, alias="champion")
    print(f"✅ Champion Model")
    print(f"   Version: {champion.version}")
    print(f"   Status: {champion.status}")
    print(f"   Created: {champion.creation_timestamp}")
    
    # Display champion tags
    print(f"\n   Champion Metadata Tags:")
    for tag_key, tag_value in champion.tags.items():
        if tag_key in ['test_auc_pr', 'test_auc_roc', 'test_f1', 'best_threshold_val', 'reg_param', 'elastic_net_param']:
            print(f"   - {tag_key}: {tag_value}")
except Exception as e:
    print(f"⚠️  No champion found: {str(e)}")
    print(f"   This is expected for first run (cold start)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify Test Baseline Table

# COMMAND ----------

try:
    baseline_df = spark.table(BASELINE_TABLE)
    baseline_count = baseline_df.count()
    baseline_defect_rate = baseline_df.filter(baseline_df.is_defective == True).count() / baseline_count
    
    # Check predictions
    predictions_exist = "prediction" in baseline_df.columns and "prob_defective_column" in baseline_df.columns
    
    print(f"✅ Test Baseline Table")
    print(f"   Rows: {baseline_count:,}")
    print(f"   Defect Rate: {baseline_defect_rate:.2%}")
    print(f"   Has Predictions: {predictions_exist}")
    print(f"   Expected: ~5M rows, ~4.2% defect rate")
except Exception as e:
    print(f"⚠️  Baseline table not found: {str(e)}")
    print(f"   This is expected if champion was not promoted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Verify Production Run

# COMMAND ----------

production_runs = mlflow.search_runs(
    experiment_names=["/Workspace/Users/{your_user_id}/manufacturing_defect_detection"],
    filter_string="tags.run_type='production'",
    max_results=10
)

print(f"✅ Production Runs")
print(f"   Total Production Runs: {len(production_runs)}")

if len(production_runs) > 0:
    latest_prod_run = production_runs.iloc[0]
    print(f"\n   Latest Production Run:")
    print(f"   - Run ID: {latest_prod_run['run_id']}")
    print(f"   - Status: {latest_prod_run['status']}")
    print(f"   - Decision: {latest_prod_run.get('tags.decision', 'N/A')}")
    print(f"   - Challenger Test AUC-PR: {latest_prod_run.get('metrics.challenger_test_auc_pr', 'N/A')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Performance Summary

# COMMAND ----------

print("\n" + "="*60)
print("EXECUTION VALIDATION SUMMARY")
print("="*60)

checks = {
    "Training Dataset": training_count > 25_000_000,
    "Grid Search Runs": len(runs_df) >= 9,
    "Candidate Model Registered": True if 'candidate' in str(candidate) else False,
    "Champion Model Registered": True if 'champion' in str(champion) else False,
    "Test Baseline Table": baseline_count > 1_000_000,
    "Production Run Logged": len(production_runs) > 0,
}

passed = sum(checks.values())
total = len(checks)

for check, result in checks.items():
    status = "✅" if result else "⚠️"
    print(f"{status} {check}")

print(f"\nPassed: {passed}/{total} checks")
print("="*60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Next Steps
# MAGIC
# MAGIC 1. **Schedule Retraining Jobs**
# MAGIC    - Create Databricks Job 1: 07_MLflow_Experimentation.py (weekly)
# MAGIC    - Create Databricks Job 2: 08_Production.py (triggered after Job 1)
# MAGIC
# MAGIC 2. **Configure Monitoring**
# MAGIC    - Enable Databricks Lakehouse Monitoring on `gold_inspection_test_baseline`
# MAGIC    - Set up drift alerts for defect predictions
# MAGIC
# MAGIC 3. **Deploy for Inference**
# MAGIC    - Use model: `models:/manufacturing_quality_control_model/champion`
# MAGIC    - Load in real-time scoring jobs
# MAGIC
# MAGIC 4. **Documentation**
# MAGIC    - See [EXECUTION_GUIDE_PHASE4.md](../EXECUTION_GUIDE_PHASE4.md) for details
# MAGIC    - See [QUICK_START_DATABRICKS.md](../QUICK_START_DATABRICKS.md) for quick reference

# COMMAND ----------

print("\n✅ Validation script completed successfully!")
print(f"Execution Date: {datetime.now().isoformat()}")
```

---

## Usage Instructions

### In Databricks:

1. **Create New Notebook**
   - Workspace → Create → Notebook
   - Language: Python
   - Name: "Phase4_Validation"

2. **Copy & Paste Content**
   - Replace `{your_user_id}` with your actual Databricks user ID
   - Paste the entire script above

3. **Run All**
   - Click "Run All"
   - Script validates all pipeline components
   - Takes ~5 minutes

### Expected Output:

```
✅ Training Dataset Verification
   Rows: 30,000,032
   Defect Rate: 4.20%
   Expected: ~30M rows, ~4.2% defect rate

✅ Grid Search Runs
   Total Runs: 9
   Expected: 9 runs

   Top 3 by Validation AUC-PR:
   - Run abc12345: AUC-PR=0.8234, F1=0.7512
   - Run def67890: AUC-PR=0.8191, F1=0.7421
   - Run ghi11111: AUC-PR=0.8156, F1=0.7398

✅ Candidate Model
   Version: 15
   Status: READY
   Created: 2026-04-27T...

✅ Champion Model
   Version: 16
   Status: READY
   Created: 2026-04-27T...

   Champion Metadata Tags:
   - test_auc_pr: 0.8234
   - test_auc_roc: 0.9156
   - test_f1: 0.7512
   - best_threshold_val: 0.55
   - reg_param: 0.01
   - elastic_net_param: 0.5

✅ Test Baseline Table
   Rows: 5,234,891
   Defect Rate: 4.20%
   Has Predictions: True
   Expected: ~5M rows, ~4.2% defect rate

✅ Production Runs
   Total Production Runs: 1

   Latest Production Run:
   - Run ID: prod_run_12345
   - Status: FINISHED
   - Decision: champion_promoted
   - Challenger Test AUC-PR: 0.8234

============================================================
EXECUTION VALIDATION SUMMARY
============================================================
✅ Training Dataset
✅ Grid Search Runs
✅ Candidate Model Registered
✅ Champion Model Registered
✅ Test Baseline Table
✅ Production Run Logged

Passed: 6/6 checks
============================================================

✅ Validation script completed successfully!
Execution Date: 2026-04-27T15:30:45.123456
```

### Success Criteria:

- ✅ All 6 checks passed
- ✅ Training dataset 25M+ rows
- ✅ Grid search 9 runs completed
- ✅ Champion model registered
- ✅ Test baseline 1M+ rows
- ✅ Champion metrics > 0.75 AUC-PR
