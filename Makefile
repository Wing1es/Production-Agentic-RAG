run-docker-compose:
	docker compose up

run-evals-retriever:
	PYTHONPATH=apps/api/src .venv/bin/python apps/api/evals/eval_retriever.py