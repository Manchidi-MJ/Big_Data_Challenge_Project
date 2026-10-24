"""
Run the full pipeline end-to-end:
    1. data cleaning  ->  2. EDA  ->  3. RFM segmentation  ->  4. churn model

If data/raw/ecommerce_churn.csv is missing, a synthetic 10K sample is
generated first so the pipeline always runs. Replace that file with the real
Kaggle CSV for the final results.

Usage:
    python -m src.run_pipeline
"""
from __future__ import annotations

import os

from . import config
from .generate_sample_data import generate


def ensure_data() -> None:
    if not os.path.exists(config.RAW_CSV):
        os.makedirs(os.path.dirname(config.RAW_CSV), exist_ok=True)
        print("Raw CSV not found -> generating synthetic 10K sample...")
        generate(rows=10000).to_csv(config.RAW_CSV, index=False)
        print(f"Synthetic data written to {config.RAW_CSV}")


def main() -> None:
    ensure_data()

    from . import pipeline_01_data_cleaning as s1
    from . import pipeline_02_eda as s2
    from . import pipeline_03_rfm as s3
    from . import pipeline_04_churn_model as s4

    print("\n" + "=" * 70 + "\nSTAGE 1: DATA CLEANING\n" + "=" * 70)
    s1.main()
    print("\n" + "=" * 70 + "\nSTAGE 2: EDA\n" + "=" * 70)
    s2.main()
    print("\n" + "=" * 70 + "\nSTAGE 3: RFM SEGMENTATION\n" + "=" * 70)
    s3.main()
    print("\n" + "=" * 70 + "\nSTAGE 4: CHURN MODELLING\n" + "=" * 70)
    s4.main()
    print("\nPipeline complete. See the outputs/ directory.")


if __name__ == "__main__":
    main()
