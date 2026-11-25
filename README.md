# ViCAP

Hệ thống xử lý theo lô cho khoá luận: thu thập video TikTok, phân loại nội dung từng video bằng mô hình P1 và dựng hồ sơ creator. Kế hoạch ở [PLAN.md](PLAN.md), danh sách việc ở [TODO.md](TODO.md).

## Cài đặt

```bash
uv sync --extra dev --extra scripts     # thêm --extra spark / gpu / airflow khi cần
cp .env.example .env                    # điền mật khẩu; không commit .env
```

## Test

```bash
uv run pytest -m "not integration"      # hoặc: make test
uv run ruff check src tests
```

Test `integration` cần dịch vụ docker-compose (`make up PROFILE=core`).

## Cấu trúc

- `src/vicap/`: `config`, `iceberg/` (catalog, reader, writer, tables), `quality/` (check PyArrow + NumPy, suite theo bảng), `storage/` (key MinIO + quy ước tên file, đọc / ghi object, quarantine), `clickhouse/`. Các package `ingestion/`, `bronze/`, `silver/`, `features/`, `model/`, `gold/`, `pipelines/` mới có khung.
- `dags/`: định nghĩa DAG Airflow, mỗi task gọi một `vicap.pipelines.run_<bước>`.
- `apps/`: `backend/` (FastAPI, `pip install .[api]`), `frontend/` (Vite + React).
- `infra/`: Dockerfile và cấu hình từng service (airflow, gpu_worker, clickhouse, kafka, superset); `docker-compose.yml` ở gốc.
- `configs/`: hashtag, lịch crawl lại.
- `scripts/`: script Giai đoạn 0 (yt-dlp). Input của phép đối chiếu nằm ở `phase0/phase0_inputs/`.
