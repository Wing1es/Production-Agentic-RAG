run-docker-compose:
	docker compose up

build:
	docker compose up --build

run-evals-retriever:
	PYTHONPATH=apps/api/src .venv/bin/python apps/api/evals/eval_retriever.py

del-cache:
	docker builder prune --filter type=exec.cache