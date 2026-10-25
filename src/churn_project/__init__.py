"""churn_project — reusable modules for the E-Commerce Customer Churn project.

Modules
-------
config              Load project_config.yaml and resolve paths.
data_loading        Read the raw CSV (pandas and PySpark).
data_validation     Schema / quality checks and the data-quality report.
preprocessing       Cleaning, type coercion, category normalisation.
pyspark_pipeline    Reproducible PySpark cleaning pipeline.
rfm_analysis        True RFM scoring and segmentation.
exploratory_analysis  EDA tables and figures.
feature_engineering Model feature preparation + leakage audit.
model_training      Build/fit sklearn pipelines (baseline, LR, RF).
model_evaluation    Metrics, curves, confusion matrices, comparison table.
reporting           Small helpers for saving tables/figures/metrics.
"""

__version__ = "0.1.0"
