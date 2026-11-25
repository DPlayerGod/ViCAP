# TODO — Xây hệ thống ViCAP

Đánh dấu `[x]` khi xong. Thứ tự từ trên xuống; mỗi giai đoạn có tiêu chí hoàn thành ở cuối. Chi tiết thiết kế xem `PLAN.md`.

## Giai đoạn 0 — Kiểm tra thu thập bằng yt-dlp ✅

- [x] Cài `yt-dlp` mới nhất (`pip install -U yt-dlp`), kiểm tra `yt-dlp --version` — 2026.11.23, trong `.venv` (`uv pip install "yt-dlp[default,curl-cffi]" opencv-python-headless`)
- [x] ~~Đăng nhập TikTok trên Chrome / Firefox để yt-dlp lấy cookie~~ — không cần: 186 + 60 bài tải được không cookie. Chrome trên Windows không đọc được cookie ("Could not copy Chrome cookie database"); nếu sau này bị chặn thì dùng `--cookies-from-browser firefox`
- [x] Viết `scripts/test_collect.py`: tải N bài gần nhất mỗi creator, đặt tên file theo PLAN mục 4.2, ghi `metadata.csv` + `report.txt` (+ `info/{video_id}.json` gốc, thử lại 2 lần khi lỗi)
- [x] Soạn `phase0/test_creators.txt` — 8 creator của dataset (khoailangthang, truongconthu04, kp_solotrip, caocuongvu, anhdaubep_vn, todidau, thaynhatdayvan, hanagianganhh); phép đối chiếu chính chạy trên 186 video cụ thể ở `phase0/phase0_inputs/phase0_urls.txt`
- [x] Chạy `python scripts/test_collect.py test_creators.txt --n 10 --out test_out/` (6 creator × 10 bài)
- [x] Đọc `test_out/report.txt`:
  - [x] Tỉ lệ video thường tải thành công — 47/48 (97,9%); bài lỗi là `status=10231` tạm thời, thử lại thì qua → đã thêm retry
  - [x] Tỉ lệ đủ trường: caption, hashtag, thời lượng, thumbnail, lượt xem / thích — video 47/47 đủ cả; slideshow 11/12 (1 bài không có nhạc nên không có thời lượng)
  - [x] Slideshow có tải được không — yt-dlp chuẩn **chỉ trả mp3** + ảnh bìa. Ảnh lấy được từ `imagePost` trong itemStruct (qua `TikTokIE._extract_web_data_and_status`, hàm nội bộ): 12/12 slideshow, 34/34 ảnh; slideshow không nhạc thì không có mp3
  - [x] Có lỗi chặn (403, captcha, danh sách rỗng) không — không có
  - [x] Thời gian trung bình mỗi video — video 2,25 s, slideshow 1,17 s (60 bài / 193 s, nghỉ 1 s giữa các bài)
- [x] Đối chiếu với dataset v5 (Apify) cùng `video_id` — `scripts/phase0_compare.py` trên 186 video (`phase0/phase0_inputs/`: `phase0_urls.txt`, `phase0_ref.csv`), không cookie, 36 phút:
  - tải 186/186 (181 video, 5 slideshow chỉ có audio)
  - caption khớp 186/186 sau khi bỏ `#hashtag` (dataset tách hashtag ra cột riêng); hashtag khớp 186/186 (yt-dlp không có `tags` → lấy `#…` trong description; bỏ ký tự zero-width space `\u200b`)
  - ảnh bìa (`cover`) **MD5 khớp 186/186** (byte-identical), kích thước 186/186; 108 PNG / 78 JPEG
  - giờ trong tên file khớp UTC 186/186
  - tên file dataset **bỏ dấu `.`** trong username (`harry.nista` → `harrynista__…`, 5 creator) → ghi ở PLAN 4.2 (`legacy_stem`)
- [x] Kiểm tra 8 ảnh (ảnh bìa + 7 frame) trích từ mp4 yt-dlp giống cách cũ:
  - `_0`: ảnh bìa `cover` lưu nguyên bytes, MD5 khớp 186/186
  - `_1…_7`: cắt theo chỉ số frame lệch dHash median 1 bit (≤ 5 bit: 89,7%), seek theo ms median 4 bit (61,4%) → tạm chốt cắt theo chỉ số frame. Sau khi có `03_src/scripts/extract_frame.py` (ffmpeg `-ss` theo `dur` container, `thumbnail(640)`, JPEG q90): **thêm cách này vào `phase0_compare.py` và đo lại**; nếu lệch ≈ 0 bit thì port nguyên script sang `src/vicap/features/` (PLAN 4.1)
- [x] Chốt múi giờ của phần `YYYYMMDD_HHMMSS` trong tên file — **UTC** (`createTime`); khớp 186/186 video dataset. Không dùng `video_id >> 32` (lệch ~10 s)
- [x] Quyết định: **yt-dlp cho toàn bộ**, không dùng Apify
  - MinIO key theo `video_id`; `legacy_stem` (`{username}__{YYYYMMDD_HHMMSS}__{video_id}`) chỉ để ghép dữ liệu v5 (PLAN 4.2)
  - Slideshow **để mở** (chưa chốt cách dựng 8 ảnh); gắn cờ `media_type` từ Bronze (PLAN 4.1)

**Xong khi:** có quyết định công cụ thu thập kèm số liệu (tỉ lệ thành công, trường thiếu, thời gian). — **Đã xong (2026-11-25).**

## Giai đoạn 1 — MVP đầu–cuối

### Khung repo (từ `hose-stock-lakehouse`, xem PLAN mục 9)
- [x] Tạo repo `ViCAP`; copy `config.py`, `utils/dates.py`, `iceberg/` (`catalog`, `reader`, `writer`), `quality/checks.py` + `result.py`, `staging/quarantine.py`, `clickhouse/client.py`, `tests/conftest.py`
- [x] Đổi tên package `stock_lakehouse` → `vicap`; xoá phần chứng khoán (`dnse_sdk`, `alerts`, `streamlit_app`, DAG cũ, `WithinDailyBand`)
- [x] Bỏ Polars: `iceberg/`, `storage/`, `quality/` dùng `pa.Table`; check viết lại trên PyArrow + NumPy
- [x] `pyproject.toml`: `requires-python >=3.11`, deps theo nhóm `airflow / spark / gpu / dev` (+ `scripts` cho Giai đoạn 0); ghim `yt-dlp[default]==2026.11.23`
- [x] `ensure_table`, `try_read_table`: bắt `NoSuchTableError` thay cho `Exception`; `create_namespace` bắt `NamespaceAlreadyExistsError`
- [x] `iceberg/writer.py`: `replace_by_keys` (`overwrite` + `In("video_id", ids)`) cho bảng có cột mảng — đã thử trên SqlCatalog (sqlite) với `text_emb` 2×768, `image_emb` 8×768
- [x] `iceberg/tables.py`: schema v0 cho 8 bảng ở PLAN mục 4
- [x] `config.py`: bỏ mật khẩu mặc định, thiếu biến môi trường thì báo lỗi khi khởi tạo config (không phải lúc import); thêm `KafkaConfig`, `MlflowConfig`, `ModelConfig`
- [x] `Makefile` thay cho script `.ps1`; sửa `.gitignore` (thêm `models/`, `data/`, sửa `.DS_Stor`); `.env.example`
- [x] `staging/` → `storage/`: `paths.py` (key MinIO theo `video_id`: `raw_video`, `frame`, `audio`, `quarantine`; dựng / tách `legacy_stem` để ghép v5), `objects.py` (`put_bytes` / `get_bytes` / `exists`), `quarantine.py` (chỉ lỗi cấp lô); bỏ staging Parquet của HOSE; theo PLAN mục 4.2 + test
- [x] Bảng `ops.quarantine` + `quality/quarantine.py` (`write_quarantine_rows`, ghi đè theo `(stage, batch_id)`) cho lỗi theo dòng + test trên SqlCatalog
- [x] `quality/`: check vector (`VectorDtype`, `VectorShape`, `NonZeroVector`, `VectorRange`), `NonEmpty`, `Covers`, `MeanShift`, `split_valid`; `suites.py` cho video theo hai cấp (PLAN mục 9.3) + test
- [ ] Xác nhận checkpoint PhoBERT của notebook v12 (đang để `vinai/phobert-base` trong `ModelConfig.text_encoder`)

### Hạ tầng
- [ ] `docker-compose.yml` (từ compose HOSE, ghim phiên bản image): profile `core` (Postgres, MinIO, Iceberg REST, ClickHouse), `stream` (Kafka), `pipeline` (Airflow + Spark local), `ml` (MLflow, image `gpu-worker`), `serving` (Superset)
- [ ] `.env.example` cho mật khẩu; `.env` thật đưa vào `.gitignore`
- [ ] Tạo bucket MinIO: `raw` (lớp 0: mp4, ảnh bìa, ảnh slideshow; chỉ crawler có quyền ghi), `lakehouse` (`derived/` frame + audio, quarantine, Iceberg), `mlflow` (artifact); không mở bucket public
- [ ] Postgres: tạo DB `airflow`, `mlflow`, `superset`
- [ ] Iceberg catalog REST (`tabulario/iceberg-rest`, JDBC trên Postgres) cho cả PyIceberg và Spark
- [ ] `infra/airflow/Dockerfile`: OpenJDK 17, `pyspark` 3.5.x, jar `iceberg-spark-runtime-3.5_2.12` + `iceberg-aws-bundle`; SparkSession có `spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions`
- [ ] `infra/gpu_worker/Dockerfile`: base CUDA + torch, `pip install .[gpu]`
- [ ] Cài NVIDIA Container Toolkit trên host; kiểm tra `docker run --gpus all … nvidia-smi`
- [ ] Airflow mount `/var/run/docker.sock`; DockerOperator có `device_requests` (GPU) và `network_mode="lakehouse_network"`
- [ ] `infra/superset/Dockerfile` cài `clickhouse-connect`
- [ ] `infra/clickhouse/init.sql` (DDL 3 bảng phục vụ), `infra/kafka/create_topics.sh` (topic `tiktok.videos`)
- [ ] `dags/.airflowignore`; compose mount `./dags` vào Airflow, build từ `./infra/<service>`
- [ ] Đo RAM khi bật toàn bộ dịch vụ; nếu thiếu thì chia docker-compose profile

### Dữ liệu ban đầu
- [ ] Viết job nạp dataset v5 (nhãn, metadata, transcript, embedding `.npz`) vào Bronze / Silver / Gold
- [ ] Kiểm tra: 2.107 video, 168 creator, `text_emb` 2×768, `image_emb` 8×768 (PLAN mục 4.1)

### Mô hình
- [ ] Tải `final_model.pt` + `kltn_model.py` của notebook v12 (`kaggle kernels output cngngquc/mainnotebook -p models/`)
- [ ] Chép `kltn_model.py` vào `src/vicap/model/`, **giữ nguyên tên module** `kltn_model`; chỉ để `final_model.pt` trong `models/`
- [ ] Chạy `register_p1.py` (log với `code_paths=[.../kltn_model.py]`) → `kltn-p1` phiên bản 1, alias `@champion`, ngưỡng 0,75
- [ ] `vicap.model.predict.predict_batch`: gán nhãn theo `decide(..., at_least_one=True)` (PLAN mục 4.3)
- [ ] Kiểm tra `predict_batch` trên 100 video Gold khớp xác suất và nhãn của notebook

### Pipeline
- [ ] `ingestion/`: crawler yt-dlp (theo kết quả Giai đoạn 0) → metadata vào Kafka topic `tiktok.videos`, mp4 / ảnh slideshow + ảnh bìa lên `raw/videos/{video_id}/` bằng `storage.objects.put_bytes`, message Kafka mang key (PLAN mục 4.2)
- [ ] `bronze/`: consumer kafka-python đọc tới `end_offsets` (so `position`), ghi đè `bronze.videos_raw` theo `batch_id`, rồi mới commit offset
- [ ] `silver/silver_clean.py` (Spark): làm sạch caption / hashtag, xác định `media_type` → `silver.videos` (`MERGE INTO` theo `video_id`)
- [ ] `features/audio.py`: tách audio như `extract_audio.py` (`-vn -ar 16000 -ac 1 -b:a 64k` → mp3), bỏ audio < 1 s → `derived/audio/`
- [ ] `features/asr.py`: PhoWhisper → `silver.transcripts`
- [ ] `features/frames.py`: 8 ảnh như `extract_frame.py` (`_0` từ `cover.*`, tất cả `thumbnail(640)` + JPEG q90) → `derived/frames/`
- [ ] `features/embed.py`: SigLIP2 + PhoBERT → `gold.features` (spec ở PLAN mục 4.1; ghi bằng `replace_by_keys`)
- [ ] `model/predict.py` trong DAG → `gold.video_predictions` (ghi bằng `replace_by_keys`)
- [ ] `gold/profile.py` (Spark) → `gold.creator_profiles`, `gold.creator_profile_history`
- [ ] Nạp 3 bảng gold sang ClickHouse
- [ ] `dags/daily_pipeline.py` nối đủ các task (mỗi task gọi một `vicap.pipelines.run_<bước>(batch_id)`); `asr`, `embed`, `predict` chạy bằng DockerOperator; `quality_check` ở cuối
- [ ] `pipelines/`: `run_<bước>(batch_id)` cho từng bước + `run_daily(batch_id)` nối tất cả để chạy không cần Airflow / test đầu–cuối
- [ ] Chạy lại cùng một lô hai lần → không có bản ghi trùng
- [ ] Lô có 1 video lỗi → video đó vào `ops.quarantine`, các video còn lại vẫn tới dashboard

### Dashboard
- [ ] Superset kết nối ClickHouse
- [ ] Biểu đồ: phân bố nhóm nội dung; bảng creator lọc theo nhóm chính và mức tin cậy; hồ sơ một creator; lịch sử hồ sơ

### Đợt B (đánh giá trên dữ liệu thật)
- [ ] Crawl Đợt B: 25–30 creator mới, 10 video gần nhất, không chọn lọc
- [ ] Chạy `daily_pipeline` trên Đợt B
- [ ] Gán nhãn tay Đợt B theo guideline v5
- [ ] Đo F1 Cấp 1 (ngưỡng 0,75, `at_least_one=True`) và độ đúng hồ sơ Cấp 2
- [ ] Đợt B2: crawl lại cùng creator → kiểm tra chỉ video mới được xử lý, hồ sơ cập nhật

**Xong khi:** một lệnh trigger DAG đưa video Đợt B từ TikTok tới dashboard; có số F1 và độ đúng hồ sơ trên Đợt B.

## Giai đoạn 2 — Mở rộng

- [ ] `configs/hashtags.yaml`: danh sách hashtag theo nhãn
- [ ] Task `discover`: tìm creator mới theo hashtag, lọc theo follower, ghi `bronze.creators_raw`
- [ ] DAG `recrawl`: crawl lại theo tần suất đăng
- [ ] Chọn cách gộp hồ sơ theo số video (dưới 10: trung bình xác suất; từ 10: đếm nhãn)
- [ ] Cờ đổi hướng nội dung (cosine hồ sơ mới so với cũ)
- [ ] `apps/backend/` FastAPI (`pip install .[api]`): `/creators` (lọc theo nhãn, số video), `/creators/{username}`, `/creators/{username}/history`, `/compare`; test bằng ClickHouse có dữ liệu v5
- [ ] `apps/frontend/` (Vite + React): màn Tìm creator, Hồ sơ creator, So sánh creator
- [ ] Thêm `backend`, `frontend` vào docker-compose profile `app` (`build: ./apps/backend`, `./apps/frontend`)

## Việc song song

- [ ] Thí nghiệm bổ sung trên OOF: luật loại trừ Others, phân tích theo loại video cho P1, hồ sơ P1 so với concat, creator 5–9 video
- [ ] Viết khóa luận: chương dữ liệu, phương pháp, kết quả
