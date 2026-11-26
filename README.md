# ViCAP

Hệ thống xử lý theo lô cho khoá luận: thu thập video TikTok, phân loại nội dung từng video bằng mô hình P1 và dựng hồ sơ creator. Kế hoạch ở [PLAN.md](PLAN.md), danh sách việc ở [TODO.md](TODO.md).

## Cài đặt

```bash
uv sync --extra dev --extra scripts --extra crawler   # thêm --extra spark / gpu khi cần ([airflow] chỉ cài trong image)
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

## Crawl

Điền username vào `configs/creators.txt` (mỗi dòng một creator), bật Kafka (`make up-crawl`), rồi:

```bash
make crawl                              # n=10 bài gần nhất mỗi creator trong configs/creators.txt
make crawl CREATORS=user_a,user_b N=2   # creator chỉ định
uv run python -m vicap.pipelines.crawl --help
```

Mỗi bài: file lên `raw/videos/{video_id}/` (`video.mp4` hoặc `img_NNN.*`, `cover.*`, `info.json`) bằng user MinIO `vicap-crawler`, rồi một message vào topic `tiktok.videos`. Bài lỗi sau 3 lần thử vẫn có message `status="error"`. TikTok chặn thì thêm `--cookies-from-browser firefox`.

## Bronze

```bash
make bronze      # tiktok.videos → bronze.videos_raw (cần make up PROFILE="core stream")
```

Đọc message mới từ offset đã commit của group `bronze-writer` tới cuối topic, đọc `info.json` từ `raw/`, khử trùng theo `(batch_id, video_id)`, ghi đè `bronze.videos_raw` theo `batch_id`, rồi mới commit offset. Chạy lại không sinh dòng trùng. Message hỏng, `info.json` không đọc được (stage `bronze_videos`) và bài crawl lỗi (stage `crawl`) vào `ops.quarantine`.

**Iceberg REST báo lỗi 500 sau khi khởi động lại Docker / `wsl --shutdown`** (`UncheckedSQLException: Failed to execute query`): Docker bật lại `iceberg-rest` trước Postgres, catalog giữ kết nối JDBC hỏng. Chạy `docker compose restart iceberg-rest`.

## Silver

```bash
make up-spark    # chỉ giữ lõi + Airflow (Spark chạy trong airflow-scheduler, driver 1 GB)
make silver      # bronze.videos_raw → silver.videos; make silver BATCH=20261125-143314 để chỉ xử lý một lô
```

Spark giữ bản crawl mới nhất mỗi `video_id`, dựng `caption` / `hashtags` đúng như dataset v5 (`vicap.silver.text`, đã so 186 video Giai đoạn 0), kiểm định `silver_videos_suite` rồi `MERGE INTO silver.videos`. Video đã có chỉ được cập nhật khi bản mới crawl sau, nên chạy lại / chạy lô cũ không đè. Dòng lỗi vào `ops.quarantine` (stage `silver_videos`).

## Media, ASR, embedding

```bash
make media       # silver.videos → 8 ảnh + mp3 ở lakehouse/derived/ + silver.media (máy host, uv sync --extra media)
make up-gpu      # giữ lõi + Airflow + docker-proxy + MLflow, tắt phần còn lại
make asr         # PhoWhisper trong gpu-worker → silver.transcripts (lần đầu tải model ~3 GB)
make embed       # PhoBERT + SigLIP2 trong gpu-worker → gold.features
make register    # một lần: models/final_model.pt → MLflow kltn-p1, alias @champion
make predict     # models:/kltn-p1@champion → gold.video_predictions
```

Cắt ảnh / tách audio là port nguyên script dựng dataset lúc train (`scripts/reference/`); ASR, PhoBERT, SigLIP2 giống notebook (đã so với cache `.npz` của Kaggle, xem TODO).

## Test

```bash
uv run pytest -m "not integration"      # hoặc: make test
uv run ruff check src tests dags
```

Test `integration` cần dịch vụ docker-compose (`make up PROFILE=core`); dịch vụ chưa chạy thì test tự bỏ qua.
Test `spark` (cần Java + jar Iceberg) chạy trong container Airflow: `make test-spark`.

## Cấu trúc

- `src/vicap/`: `config`, `iceberg/` (catalog, reader, writer, tables), `quality/` (check PyArrow + NumPy, suite theo bảng), `storage/` (key MinIO + quy ước tên file, đọc / ghi object, quarantine), `clickhouse/`. `ingestion/` (crawler yt-dlp: `parse`, `message` Kafka, `tiktok`, `crawler`), `bronze/` (Kafka → Bronze), `silver/` (`text`: caption / hashtag như v5, `videos`: Spark), `pipelines/{crawl,bronze,silver}`. Các package `features/`, `model/`, `gold/` mới có khung.
- `dags/`: định nghĩa DAG Airflow, mỗi task gọi một `vicap.pipelines.run_<bước>`.
- `apps/`: `backend/` (FastAPI, `pip install .[api]`), `frontend/` (Vite + React).
- `infra/`: Dockerfile và cấu hình từng service (airflow, gpu_worker, postgres, minio, clickhouse, kafka, mlflow, superset); `docker-compose.yml` ở gốc.
- `configs/`: `creators.txt` (creator cần crawl); sau này hashtag, lịch crawl lại.
- `scripts/`: script Giai đoạn 0 (yt-dlp). Input của phép đối chiếu nằm ở `phase0/phase0_inputs/`.
