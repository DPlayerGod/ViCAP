# ViCAP

Hệ thống xử lý theo lô cho khoá luận: thu thập video TikTok, phân loại nội dung từng video bằng mô hình P1 và dựng hồ sơ creator. Kế hoạch ở [PLAN.md](PLAN.md), danh sách việc ở [TODO.md](TODO.md).

## Cài đặt

```bash
uv sync --extra dev --extra scripts     # thêm --extra spark / gpu / airflow khi cần
cp .env.example .env                    # điền mật khẩu; không commit .env
```

## Chạy hạ tầng

Cần Docker Desktop. Trên Windows:

- Cấp 8 GB RAM cho WSL và cho WSL trả RAM thừa về Windows: thêm vào mục `[wsl2]` của `%UserProfile%\.wslconfig` hai dòng `memory=8GB`, `autoMemoryReclaim=gradual` (mục `[experimental]` trên bản WSL cũ), rồi `wsl --shutdown`. Bật theo từng bước (`make up-spark`, `make up-gpu`, …) thì 8 GB là đủ.
- Nếu WSL dùng `networkingMode=mirrored`: `localhost` ra `::1` trước và kết nối vào Docker treo ~21 s (PyArrow S3 lỗi luôn). Cho Windows ưu tiên IPv4 (PowerShell Admin, đảo lại được):
  `netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4`

```bash
cp .env.example .env                    # điền mọi ô trống (mật khẩu chỉ chữ + số)
make build                              # image airflow, gpu-worker, mlflow, superset
make up PROFILE="core stream"           # MinIO, Iceberg REST, ClickHouse, Postgres, Kafka
make up PROFILE="core stream pipeline ml"
make up-all                             # cả Superset
make ps                                 # trạng thái; make stats để xem RAM
```

| Dịch vụ | Địa chỉ |
|---|---|
| Airflow | http://localhost:8080 (`AIRFLOW_ADMIN_USERNAME` / `AIRFLOW_ADMIN_PASSWORD`) |
| MinIO console | http://localhost:9001 (user root trong `.env`) |
| Iceberg REST | http://localhost:8181 |
| ClickHouse | http://localhost:8123, native 9002 |
| Kafka | localhost:29092 |
| MLflow | http://localhost:5000 |
| Superset | http://localhost:8088 (`admin` / `SUPERSET_ADMIN_PASSWORD`) |

Bật theo bước pipeline để tiết kiệm RAM (dừng các dịch vụ bước đó không cần):

| Lệnh | Giữ chạy (ngoài Postgres, MinIO, Iceberg REST, Airflow) | RAM đo được |
|---|---|---|
| `make up-crawl` | Kafka | |
| `make up-spark` | — (Spark chạy trong `airflow-scheduler`) | ~1,4 GB nghỉ, ~2,5 GB lúc Spark chạy |
| `make up-gpu` | `docker-proxy`, MLflow (+ container gpu-worker ~3 GB khi chạy) | |
| `make up-serving` | ClickHouse, Superset | |

Windows chưa có `make`: `winget install ezwinports.make` (hoặc chạy lệnh `docker compose` tương ứng trong Makefile).

Kiểm tra: trigger DAG `infra_smoke` (Spark `MERGE INTO` trên Iceberg, PyIceberg + MinIO, GPU) và `make test-integration`.

## Test

```bash
uv run pytest -m "not integration"      # hoặc: make test
uv run ruff check src tests dags
```

Test `integration` cần dịch vụ docker-compose (`make up PROFILE=core`); dịch vụ chưa chạy thì test tự bỏ qua.

## Cấu trúc

- `src/vicap/`: `config`, `iceberg/` (catalog, reader, writer, tables), `quality/` (check PyArrow + NumPy, suite theo bảng), `storage/` (key MinIO + quy ước tên file, đọc / ghi object, quarantine), `clickhouse/`. Các package `ingestion/`, `bronze/`, `silver/`, `features/`, `model/`, `gold/`, `pipelines/` mới có khung.
- `dags/`: định nghĩa DAG Airflow, mỗi task gọi một `vicap.pipelines.run_<bước>`.
- `apps/`: `backend/` (FastAPI, `pip install .[api]`), `frontend/` (Vite + React).
- `infra/`: Dockerfile và cấu hình từng service (airflow, gpu_worker, postgres, minio, clickhouse, kafka, mlflow, superset); `docker-compose.yml` ở gốc.
- `configs/`: hashtag, lịch crawl lại.
- `scripts/`: script Giai đoạn 0 (yt-dlp). Input của phép đối chiếu nằm ở `phase0/phase0_inputs/`.
