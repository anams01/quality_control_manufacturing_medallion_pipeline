# 🚀 Quick Start: Run ML Pipeline in Databricks
## Manufacturing Quality Control - Defect Detection

---

## ⚡ TL;DR - Execute in 3 Steps

### Step 1: Verify Prerequisites (5 minutes)
```sql
-- Run in Databricks SQL
SELECT COUNT(*) FROM workspace.ana_martin17.gold_inspection_training_dataset;
-- Expected: ~30,000,032 rows
```

### Step 2: Run Grid Search (45-90 minutes)
1. Open notebook: `notebooks/07_MLflow_Experimentation.py`
2. Click **Run All**
3. Wait for 9 training runs to complete
4. Check MLflow experiment for best candidate (highest validation AUC-PR)

### Step 3: Run Production Pipeline (30-60 minutes)
1. Open notebook: `notebooks/08_Production.py`
2. Click **Run All**
3. Wait for production evaluation to complete
4. Verify champion model registered in Unity Catalog

---

## 📍 File Locations in Databricks

Navigate to **Workspace** → **Users** → **[your user ID]** → **quality_control_manufacturing_medallion_pipeline**

```
notebooks/
├── 07_MLflow_Experimentation.py  ← Grid search (Step 2)
├── 08_Production.py             ← Production promotion (Step 3)
├── 08_Utils.py                  ← Production utilities
├── 07_Utils.py                  ← Training utilities
└── ... (other supporting notebooks)
```

---

## 🔍 Pre-Check Commands

Run these in a Databricks cell to verify setup:

```python
# Check 1: Training dataset exists
spark.table("workspace.ana_martin17.gold_inspection_training_dataset").count()

# Check 2: Feature tables exist
spark.table("workspace.ana_martin17.gold_machine_agg_1h").count()
spark.table("workspace.ana_martin17.gold_machine_agg_24h").count()

# Check 3: MLflow configured
import mlflow
mlflow.set_experiment("/Workspace/Users/<YOUR_USER_ID>/manufacturing_defect_detection")
print(f"Experiment set to: {mlflow.get_experiment_by_name(...)}")

# Check 4: Unity Catalog access
spark.sql("SHOW TABLES IN workspace.ana_martin17").display()
```

---

## ⏱️ Execution Timeline

| Step | Notebook | Duration | Action |
|------|----------|----------|--------|
| 1 | Pre-check | 5 min | Run verification commands |
| 2 | 07_MLflow_Experimentation | 45-90 min | Run All → Wait |
| 3 | 08_Production | 30-60 min | Run All → Wait |
| 4 | Validation | 5 min | Run verification SQL |

**Total Time**: ~90-160 minutes (1.5-2.5 hours)

---

## ✅ Success Signals

### After Step 2 (Grid Search)
```
✅ 9 training runs completed
✅ Best model has validation AUC-PR > 0.75
✅ Model version N registered with alias "candidate"
✅ MLflow experiment shows all runs
```

Navigate to: **Experiments** → **manufacturing_defect_detection** → Click best run

### After Step 3 (Production)
```
✅ Challenger evaluated on test set
✅ Test AUC-PR captured and logged
✅ Champion promoted (or held if not first run)
✅ Test baseline table created
✅ All metadata tags applied
```

Navigate to: **Models** → search for `manufacturing_quality_control_model` → Click latest version

---

## 📊 Performance Dashboard

Check these after execution:

### In Databricks UI:
1. **Experiments Tab**
   - Path: **Experiments** → **manufacturing_defect_detection**
   - View all 9 grid search runs with metrics

2. **Models Tab**
   - Path: **Workspace Objects** → **Models** → `workspace.ana_martin17.manufacturing_quality_control_model`
   - View model versions and aliases

3. **Data Tab (Monitoring)**
   - Path: **Data** → **quality_control_manufacturing_medallion_pipeline** → **gold_inspection_test_baseline**
   - View baseline for Lakehouse Monitoring setup

### In SQL:
```sql
-- View champion model metadata
SELECT tag_key, tag_value 
FROM system.models.model_version_tags 
WHERE name = 'manufacturing_quality_control_model'
AND schema_name = 'ana_martin17'
ORDER BY tag_key;

-- View test baseline
SELECT COUNT(*), 
       ROUND(AVG(is_defective), 4) as defect_rate,
       COUNT(DISTINCT unit_id) as units
FROM workspace.ana_martin17.gold_inspection_test_baseline;
```

---

## 🚨 Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Notebook not found" | Path: `Workspace/Users/{user_id}/quality_control_manufacturing_medallion_pipeline/notebooks/` |
| "Table not found" | Run 05_training_dataset_generation.py first to create gold_inspection_training_dataset |
| "Experiment not found" | Create experiment: `mlflow.set_experiment("/Workspace/Users/{user_id}/manufacturing_defect_detection")` |
| "Model registry error" | Ensure using "databricks-uc" registry: `mlflow.set_registry_uri("databricks-uc")` |
| "Job timeout" | Increase cluster size or increase timeout in 08_Utils.py |

---

## 📞 Key Contacts

For issues:
1. Check notebook cell output for error details
2. Review Databricks job logs: **Jobs** → Select job → **Run** → **Logs**
3. Check MLflow run details: **Experiments** → Click run → Review parameters and metrics

---

## 🎯 Next Steps After Success

1. **Schedule Regular Retraining**
   - Create Databricks Job 1: 07_MLflow_Experimentation (weekly)
   - Create Databricks Job 2: 08_Production (triggered after Job 1)

2. **Set Up Monitoring**
   - Enable Lakehouse Monitoring for `gold_inspection_test_baseline`
   - Configure drift alerts

3. **Deploy for Inference**
   - Use champion model: `models:/manufacturing_quality_control_model/champion`
   - Load in real-time scoring pipeline

---

## 📋 Execution Checklist

- [ ] Verified training dataset exists (~30M rows)
- [ ] MLflow experiment configured
- [ ] Cluster sized for ~45-90 min grid search
- [ ] Read through EXECUTION_GUIDE_PHASE4.md for details
- [ ] Step 2: Grid search started and running
- [ ] Step 2: Best candidate identified (AUC-PR > 0.75)
- [ ] Step 3: Production pipeline started
- [ ] Step 3: Champion promoted to model registry
- [ ] Validation queries executed successfully
- [ ] Performance metrics within expected ranges

---

**Status**: 🟢 READY TO EXECUTE IN DATABRICKS

All notebooks adapted and verified for manufacturing quality control defect detection pipeline.
