.PHONY: up down logs test test-integration lint dag-trigger

PROFILE ?= core
DAG ?= daily_pipeline

# make up PROFILE=core | make up PROFILE="core stream pipeline"
up:
	docker compose $(foreach p,$(PROFILE),--profile $(p)) up -d

down:
	docker compose --profile "*" down

logs:
	docker compose $(foreach p,$(PROFILE),--profile $(p)) logs -f --tail=200

test:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests

dag-trigger:
	docker compose exec airflow-scheduler airflow dags trigger $(DAG)
