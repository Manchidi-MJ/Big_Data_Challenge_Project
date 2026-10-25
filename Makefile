# Convenience targets. On Windows, run the underlying python commands directly
# if you don't have `make` (see README).

.PHONY: install data-check run run-nospark test notebooks pptx clean

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

run:
	python -m churn_project.run_all

run-nospark:
	python -m churn_project.run_all --no-spark

test:
	python -m pytest

notebooks:
	python notebooks/_build_notebooks.py

pptx:
	python presentation/build_pptx.py

clean:
	python -c "import shutil,os; [shutil.rmtree(p,ignore_errors=True) for p in ['outputs/figures','outputs/tables','outputs/metrics','outputs/models']]; [os.makedirs(p,exist_ok=True) for p in ['outputs/figures','outputs/tables','outputs/metrics','outputs/models']]"
