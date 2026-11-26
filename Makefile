.PHONY: up up-all down build ps stats logs test test-integration lint dag-trigger \
        up-crawl up-spark up-gpu up-serving crawl bronze silver media asr embed register predict test-spark

PROFILE ?= core
DAG ?= daily_pipeline
N ?= 10
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

# Crawl từ máy host (cần make up-crawl): make crawl | make crawl CREATORS=a,b N=2
crawl:
	uv run python -m vicap.pipelines.crawl --n $(N) $(if $(CREATORS),--creators $(CREATORS))

# Kafka → bronze.videos_raw từ máy host (cần make up PROFILE="core stream")
bronze:
	uv run python -m vicap.pipelines.bronze

# Spark chạy trong airflow-scheduler (cần make up-spark): make silver | make silver BATCH=20261125-143314
silver:
	docker compose exec airflow-scheduler python -m vicap.pipelines.silver $(if $(BATCH),--batch-id $(BATCH))

# 8 ảnh + audio → lakehouse/derived/ + silver.media, chạy trên máy host (uv sync --extra media)
media:
	uv run python -m vicap.pipelines.media $(if $(BATCH),--batch-id $(BATCH))

# Chạy một bước trong gpu-worker như DockerOperator; mount ./src để không phải build lại image khi sửa code
GPU_RUN = $(DC) run --rm --no-deps -v ./src:/app/src -e PYTHONPATH=/app/src gpu-worker

# PhoWhisper → silver.transcripts trong gpu-worker (cần make up-gpu). Mount ./src để không phải build lại
# image mỗi lần sửa code; lần đầu tải model (~3 GB) vào volume vicap_hf_cache.
asr:
	$(GPU_RUN) python -m vicap.pipelines.asr $(if $(BATCH),--batch-id $(BATCH))

# PhoBERT + SigLIP2 → gold.features trong gpu-worker (cần make up-gpu)
embed:
	$(GPU_RUN) python -m vicap.pipelines.embed $(if $(BATCH),--batch-id $(BATCH))

# Đăng ký models/final_model.pt vào MLflow: kltn-p1, alias @champion (chạy lại không đăng ký trùng)
register:
	$(DC) run --rm --no-deps -v ./src:/app/src -v ./models:/p1:ro -e PYTHONPATH=/app/src gpu-worker python -m vicap.model.register_p1 --checkpoint /p1/final_model.pt

# models:/kltn-p1@champion trên gold.features → gold.video_predictions
predict:
	$(GPU_RUN) python -m vicap.pipelines.predict $(if $(BATCH),--batch-id $(BATCH))

# Test Spark (marker spark) trong airflow-scheduler: chép đè tests/ vào /tmp/vicap-test, cài pytest vào venv của image
# (mất khi tạo lại container)
test-spark:
	docker compose exec -T airflow-scheduler mkdir -p /tmp/vicap-test
	docker compose cp tests airflow-scheduler:/tmp/vicap-test/
	docker compose cp pyproject.toml airflow-scheduler:/tmp/vicap-test/pyproject.toml
	docker compose exec -T airflow-scheduler sh -c "cd /tmp/vicap-test && python -m pip install -q pytest && python -m pytest -m spark"

dag-trigger:
	docker compose exec airflow-scheduler airflow dags trigger $(DAG)
