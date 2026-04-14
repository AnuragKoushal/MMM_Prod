.PHONY: install run test lint format clean

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest

test-fast:
	pytest -x -q --no-header --cov=src --cov-report=term-missing

lint:
	ruff check src tests app.py

format:
	ruff format src tests app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage outputs/*.json outputs/*.csv outputs/*.pkl
