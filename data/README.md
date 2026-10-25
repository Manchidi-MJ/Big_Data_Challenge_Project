# Data directory

This folder holds the project data. **Raw and processed data are not committed
to Git** (see `.gitignore`) — only the folder structure is tracked.

```
data/
├── raw/        # original, unmodified dataset (input)
├── interim/    # intermediate artefacts (optional)
└── processed/  # cleaned data + RFM output produced by the pipeline
```

## Dataset

**E-Commerce Customer Churn 2025 – 10K**

| Field | Value |
|---|---|
| Source | Kaggle |
| URL | <https://www.kaggle.com/datasets/mohammadmaaz052/e-commerce-customer-churn-2025-10k> |
| Author | Mohammad Maaz (`mohammadmaaz052`) |
| Licence | **MIT** (per Kaggle API metadata) |
| Size on Kaggle | 207,128 bytes (zipped) |
| File | `ecommerce_customer_churn_10k.csv` (≈ 950 KB unzipped) |
| Rows × columns | **10,000 × 15** (verified) |
| Target | `churned` (1 = churned, 0 = retained) |
| Provenance (real vs synthetic) | The author provides no description. Realistic/synthetic status **could not be confirmed**. |

> The verified column list and meanings are in
> [`reports/data_dictionary.md`](../reports/data_dictionary.md).

## How to obtain the dataset

### Option A — Manual download (no credentials needed)

1. Open the Kaggle page above and sign in.
2. Click **Download** to get the zip.
3. Unzip it and place `ecommerce_customer_churn_10k.csv` at:
   ```
   data/raw/ecommerce_customer_churn_10k.csv
   ```

### Option B — Kaggle API (optional)

1. Install the CLI: `pip install kaggle`
2. Put your `kaggle.json` token in `~/.kaggle/` (or set `KAGGLE_USERNAME` /
   `KAGGLE_KEY`, see `.env.example`). **Never commit this token.**
3. Run:
   ```bash
   kaggle datasets download -d mohammadmaaz052/e-commerce-customer-churn-2025-10k -p data/raw --unzip
   ```

The pipeline reads from `data/raw/ecommerce_customer_churn_10k.csv` (configurable
in `config/project_config.yaml`). No code changes are needed once the file is in
place.

## Reproducibility note

The cleaning pipeline reads only from `data/raw/` and writes to
`data/processed/`. It never edits the raw file in place, so the full pipeline can
be re-run from scratch at any time.
