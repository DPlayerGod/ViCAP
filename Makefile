.PHONY: up up-all down build ps stats logs test test-integration lint dag-trigger \
        up-crawl up-spark up-gpu up-serving

PROFILE ?= core
DAG ?= daily_pipeline
ALL_PROFILES := core stream pipeline ml serving
PROFILE_FLAGS = $(foreach p,$(PROFILE),--profile $(p))
DC := docker compose --profile "*"

# ── Bật theo profile ──
# make up PROFILE=core | make up PROFILE="core stream pipeline ml"
# pipeline cần core + stream; ml và serving cần core.
up:
	docker compose $(PROFILE_FLAGS) up -d

up-all:
	docker compose $(foreach p,$(ALL_PROFILES),--profile $(p)) up -d

# ── Bật theo bước pipeline (tiết kiệm RAM: chỉ giữ dịch vụ bước đó cần, tắt phần còn lại) ──
# Lõi luôn cần: Postgres (DB Airflow + catalog Iceberg), MinIO, Iceberg REST.
# Airflow 3: task gọi Execution API của api-server → api-server luôn phải chạy cùng scheduler.
CORE_SVCS    := postgres minio minio-init iceberg-rest
AIRFLOW_SVCS := airflow-init airflow-api-server airflow-scheduler airflow-dag-processor
OPTIONAL_SVCS := kafka kafka-init docker-proxy mlflow clickhouse superset superset-init

# $(call only,<dịch vụ cần>): bật danh sách đó, dừng các dịch vụ tuỳ chọn không có trong danh sách
define only
	$(DC) stop $(filter-out $(1),$(OPTIONAL_SVCS))
	$(DC) up -d $(1)
endef

# crawl + Bronze: Kafka
up-crawl:
	$(call only,$(CORE_SVCS) $(AIRFLOW_SVCS) kafka kafka-init)

# Spark (silver_clean, profile) chạy trong airflow-scheduler: chỉ cần lõi
up-spark:
	$(call only,$(CORE_SVCS) $(AIRFLOW_SVCS))

# asr / embed / predict: DockerOperator qua docker-proxy, predict đọc model từ MLflow
up-gpu:
	$(call only,$(CORE_SVCS) $(AIRFLOW_SVCS) docker-proxy mlflow)

# nạp ClickHouse + xem dashboard
up-serving:
	$(call only,$(CORE_SVCS) $(AIRFLOW_SVCS) clickhouse superset-init superset)

down:
	$(DC) down

# build image tự dựng: airflow, gpu-worker, mlflow, superset
build:
	$(DC) build

ps:
	$(DC) ps -a

stats:
	docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"

logs:
	docker compose $(PROFILE_FLAGS) logs -f --tail=200

test:
	uv run pytest -m "not integration"

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests dags

dag-trigger:
	docker compose exec airflow-scheduler airflow dags trigger $(DAG)
