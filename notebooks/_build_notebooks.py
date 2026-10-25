"""Generate the four project notebooks as valid .ipynb files.

Run from project root:  python notebooks/_build_notebooks.py
Each notebook wraps the reusable src modules so narrative + real outputs stay in
sync with the code. This builder is a convenience; the notebooks it writes are
the actual deliverables.
"""
from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

BOOTSTRAP = (
    "# Make the src package importable whether or not it is pip-installed.\n"
    "import sys, os\n"
    "ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) "
    "if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\n"
    "SRC = os.path.join(ROOT, 'src')\n"
    "if SRC not in sys.path:\n    sys.path.insert(0, SRC)\n"
    "os.chdir(ROOT)\n"
    "print('project root:', ROOT)"
)


def nb01():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 01 — Data Understanding & Quality Assessment\n\n"
            "**Owner:** Mkhawana Thamelia Hleketani (Data Engineer / Big Data)\n\n"
            "Loads the verified Kaggle dataset and profiles it: shape, dtypes, "
            "missing values, duplicates, invalid ranges, target balance and "
            "feature redundancy. Nothing is deleted here — this step only reports."
        ),
        new_code_cell(BOOTSTRAP),
        new_code_cell(
            "from churn_project import data_loading, data_validation as dv\n"
            "df = data_loading.load_raw_pandas()\n"
            "print('shape:', df.shape)\n"
            "df.head()"
        ),
        new_markdown_cell("## Schema check and dtypes"),
        new_code_cell(
            "print(dv.validate_schema(df))\n"
            "df.dtypes"
        ),
        new_markdown_cell("## Missing values and duplicates"),
        new_code_cell(
            "display(dv.missing_value_report(df))\n"
            "print(dv.duplicate_report(df))"
        ),
        new_markdown_cell("## Invalid values and numeric summary"),
        new_code_cell(
            "display(dv.invalid_value_report(df))\n"
            "display(dv.numeric_summary(df))"
        ),
        new_markdown_cell("## Target distribution and redundancy"),
        new_code_cell(
            "display(dv.target_distribution(df))\n"
            "display(dv.redundancy_report(df))"
        ),
        new_markdown_cell(
            "**Findings:** 10,000 x 15, one row per customer, no missing values, "
            "no duplicates. Target 37.7% churn. `total_spent_usd` and "
            "`avg_order_value` are redundant (r=0.85)."
        ),
    ]
    return nb


def nb02():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 02 — Data Cleaning (PySpark)\n\n"
            "**Owner:** Mkhawana Thamelia Hleketani\n\n"
            "Runs the reproducible PySpark cleaning pipeline and shows before/after "
            "counts. PySpark here demonstrates a *scalable design* — 10k rows do "
            "not require distributed computing."
        ),
        new_code_cell(BOOTSTRAP),
        new_markdown_cell("## Run the PySpark cleaning pipeline"),
        new_code_cell(
            "from churn_project.pyspark_pipeline import run_pipeline\n"
            "log = run_pipeline()\n"
            "print('before:', log['before'])\n"
            "print('after :', log['after'])\n"
            "print('parquet:', log['parquet_status'])\n"
            "print('csv    :', log['csv_path'])"
        ),
        new_markdown_cell("## Load the cleaned output"),
        new_code_cell(
            "import pandas as pd\n"
            "clean = pd.read_csv(log['csv_path'])\n"
            "print(clean.shape)\n"
            "clean.head()"
        ),
        new_markdown_cell(
            "**Note:** on Windows the native Parquet writer needs winutils/HADOOP_HOME; "
            "when unavailable the pipeline writes the canonical cleaned CSV and logs it."
        ),
    ]
    return nb


def nb03():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 03 — Exploratory Analysis & RFM Segmentation\n\n"
            "**Owner:** Burgos Carrera, Maria de Los Angeles (Data Analyst)\n\n"
            "EDA on churn and feature relationships, then true RFM segmentation."
        ),
        new_code_cell(BOOTSTRAP),
        new_code_cell(
            "import pandas as pd\n"
            "from churn_project import exploratory_analysis as eda, rfm_analysis as rfm\n"
            "from churn_project.config import CONFIG\n"
            "df = pd.read_csv(CONFIG.path('clean_csv'))\n"
            "print(eda.churn_overview(df))"
        ),
        new_markdown_cell("## Numeric features by churn"),
        new_code_cell("eda.grouped_numeric_summary(df)"),
        new_markdown_cell("## Generate and display EDA figures"),
        new_code_cell(
            "res = eda.run_full_eda(df)\n"
            "from IPython.display import Image, display\n"
            "for name, path in res['figures'].items():\n"
            "    print(name); display(Image(path))"
        ),
        new_markdown_cell("## RFM segmentation"),
        new_code_cell(
            "rfm_df, meta = rfm.compute_rfm(df)\n"
            "print('bins used:', meta['n_bins_used'])\n"
            "rfm.segment_summary(rfm_df)"
        ),
        new_markdown_cell(
            "**Findings:** churn is explained almost entirely by recency. "
            "'Needs Attention' and 'At Risk' segments carry ~94% churn; recent "
            "segments ~0%."
        ),
    ]
    return nb


def nb04():
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(
            "# 04 — Churn Prediction (Leakage Audit + Modelling)\n\n"
            "**Owner:** Manchidi Magaseng John (Machine Learning)\n\n"
            "Mandatory leakage audit, then two modelling experiments: A (all "
            "features) and B (leakage-resistant, primary)."
        ),
        new_code_cell(BOOTSTRAP),
        new_code_cell(
            "import pandas as pd\n"
            "from churn_project import feature_engineering as fe, model_training as mt, "
            "model_evaluation as me\n"
            "from churn_project.config import CONFIG\n"
            "df = pd.read_csv(CONFIG.path('clean_csv'))"
        ),
        new_markdown_cell("## Leakage audit"),
        new_code_cell(
            "audit = fe.audit_leakage(df)\n"
            "import json; print(json.dumps(audit, indent=2, default=str))"
        ),
        new_markdown_cell("## Two experiments: fit and evaluate"),
        new_code_cell(
            "rows = []\n"
            "for exp in ('A', 'B'):\n"
            "    X, y, feats = fe.build_xy(df, exp)\n"
            "    Xtr, Xte, ytr, yte = mt.split_data(X, y)\n"
            "    for name, pipe in mt.build_models(feats['numeric'], feats['categorical']).items():\n"
            "        pipe.fit(Xtr, ytr)\n"
            "        r = me.evaluate_model(name, pipe, Xte, yte); r['experiment'] = exp\n"
            "        rows.append(r)\n"
            "comp = pd.DataFrame(rows).sort_values(['experiment','f1'], ascending=[True, False])\n"
            "comp"
        ),
        new_markdown_cell(
            "**Findings:** Experiment A ~100% (leakage). Experiment B is near the "
            "majority baseline (ROC-AUC ~0.54), so no non-recency behaviour predicts "
            "churn. The leakage-resistant model is the honest, primary result."
        ),
        new_markdown_cell("## Full orchestration (optional)\n"
                          "Run the whole project end-to-end and persist all artefacts:"),
        new_code_cell(
            "# from churn_project.run_all import main\n"
            "# summary = main(use_spark=True)\n"
            "# summary"
        ),
    ]
    return nb


def main():
    builders = {
        "01_data_understanding.ipynb": nb01,
        "02_data_cleaning_pyspark.ipynb": nb02,
        "03_exploratory_analysis_rfm.ipynb": nb03,
        "04_churn_prediction.ipynb": nb04,
    }
    for fname, fn in builders.items():
        path = os.path.join(HERE, fname)
        nbf.write(fn(), path)
        print("wrote", path)


if __name__ == "__main__":
    main()
