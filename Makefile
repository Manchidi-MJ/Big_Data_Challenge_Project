.PHONY: install data run clean-all stage1 stage2 stage3 stage4

install:
	pip install -r requirements.txt

data:
	python -m src.generate_sample_data --rows 10000

run:
	python -m src.run_pipeline

stage1:
	python -m src.pipeline_01_data_cleaning

stage2:
	python -m src.pipeline_02_eda

stage3:
	python -m src.pipeline_03_rfm

stage4:
	python -m src.pipeline_04_churn_model

clean-all:
	rm -rf data/processed/* outputs/figures/* outputs/tables/* outputs/model_metrics.json
