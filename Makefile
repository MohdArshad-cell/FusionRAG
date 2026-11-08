.PHONY: install test run seed benchmark evaluate cli docker-up docker-down

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

seed:
	python scripts/seed_data.py

benchmark:
	python scripts/benchmark.py

evaluate:
	python scripts/evaluate_rag.py

cli:
	python scripts/cli_chat.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
