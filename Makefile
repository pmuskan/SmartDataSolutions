.PHONY: setup ingest eval test lint app report clean

setup:
	pip install -r requirements.txt

ingest:
	python -m src.ingest.pipeline

eval:
	python -m eval.run_eval
	python -m eval.ablations

test:
	pytest -v --cov=src

lint:
	ruff check src tests eval

app:
	streamlit run app.py

report:
	python report/build_report.py
	python report/build_solution_overview.py

clean:
	rm -rf data/processed data/chroma_db data/bm25_index.pkl eval/results/*.json eval/results/*.md report/*.pdf
