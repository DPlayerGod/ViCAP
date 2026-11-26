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
- [x] Xác nhận checkpoint PhoBERT của notebook v12: `vinai/phobert-base` (`TEXT_MODEL_NAME`, pyvi tách từ, `max_length=256`, mean-pool theo mask), khớp `ModelConfig.text_encoder`

### Hạ tầng
- [x] `docker-compose.yml` (từ compose HOSE, ghim phiên bản image, bí mật `${VAR:?}`, `mem_limit`): profile `core` (Postgres, MinIO, Iceberg REST, ClickHouse), `stream` (Kafka), `pipeline` (Airflow 3.4 + Spark local + `docker-proxy`), `ml` (MLflow, image `gpu-worker`), `serving` (Superset)
- [x] MinIO: image Chainguard (image chính thức bị gỡ 10/2025), ghim digest
- [x] `.env.example` cho mật khẩu; `.env` thật đưa vào `.gitignore`; địa chỉ host dùng `localhost`; cổng compose chỉ mở trên loopback 127.0.0.1
- [x] Windows + WSL `networkingMode=mirrored`: `localhost` → `::1` trước, kết nối ::1 vào Docker treo ~21 s (PyArrow S3 lỗi luôn) → chạy (Admin) `netsh interface ipv6 set prefixpolicy ::ffff:0:0/96 60 4` để ưu tiên IPv4
- [x] Tạo bucket MinIO: `raw` (lớp 0: mp4, ảnh bìa, ảnh slideshow; chỉ crawler có quyền ghi), `lakehouse` (`derived/` frame + audio, quarantine, Iceberg), `mlflow` (artifact); không mở bucket public
- [x] User MinIO `vicap-crawler` (ghi `raw`) / `vicap-pipeline` (đọc `raw`, ghi `lakehouse` + `mlflow`) + `crawler_minio_config()`; đã thử quyền bằng `mc` và `tests/test_integration_infra.py`
- [x] Postgres: tạo DB `airflow`, `iceberg`, `mlflow`, `superset` (`infra/postgres/init.sh`)
- [x] Iceberg catalog REST (`tabulario/iceberg-rest:1.6.0`, JDBC trên Postgres) cho cả PyIceberg và Spark; test `ops.quarantine` qua REST thật
- [x] `infra/airflow/Dockerfile`: Airflow 3.4.0 (Debian 13: JRE 17 chép từ `eclipse-temurin:17-jre`), OpenJDK 17, `pyspark` 3.5.x, jar `iceberg-spark-runtime-3.5_2.12` + `iceberg-aws-bundle` 1.6.1; `vicap.spark.get_spark` có `spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions` (`spark_conf` + test xong)
- [x] `infra/gpu_worker/Dockerfile`: base `pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime` (trước là 2.5.1 / cu121, đổi vì transformers chặn torch.load trên torch < 2.6), `pip install .[gpu]` (mlflow ghim 2.22.1 cho khớp server)
- [x] Kiểm tra GPU trong container (RTX 3050 Laptop, `torch 2.5.1+cu121`): Docker Desktop (WSL2) tự có GPU, không cần NVIDIA Container Toolkit; `docker compose --profile ml run --rm gpu-worker`
- [x] DockerOperator qua `docker-proxy` (không mount socket vào Airflow): `dags/common/gpu.py` có `device_requests` (GPU), `network_mode="lakehouse_network"`, volume `vicap_hf_cache`; DAG `infra_smoke` chạy xanh
- [x] `infra/superset/Dockerfile` cài `clickhouse-connect`; Superset kết nối được ClickHouse (`clickhousedb://…/vicap`)
- [x] `infra/clickhouse/init.sql` (DDL 3 bảng phục vụ), `infra/kafka/create_topics.sh` (topic `tiktok.videos`: 3 partition, retention 7 ngày, chạy lại được)
- [x] `dags/.airflowignore`; compose mount `./dags` vào Airflow, build từ `./infra/<service>`
- [x] Đặt `memory=8GB` trong `%UserProfile%\.wslconfig` rồi `wsl --shutdown` (VM Docker mặc định 7,4 GB → `docker info` MemTotal 8,33 GB); chưa bật `autoMemoryReclaim`
- [x] Đo RAM khi bật toàn bộ dịch vụ (2026-11-25, nghỉ): tổng ≈ 3,0 GB — scheduler 0,6, ClickHouse 0,54, MLflow 0,44, api-server 0,32, Kafka 0,3, dag-processor 0,25, Superset 0,2, Iceberg REST 0,14, MinIO 0,1, Postgres 0,07 GB. Lúc chạy cộng thêm Spark (~1–2 GB, trong scheduler) và gpu-worker (~3 GB) → sát 7,4 GB của VM

### Dữ liệu ban đầu
- [ ] Viết job nạp dataset v5 (nhãn, metadata, transcript, embedding `.npz`) vào Bronze / Silver / Gold
- [ ] Kiểm tra: 2.107 video, 168 creator, `text_emb` 2×768, `image_emb` 8×768 (PLAN mục 4.1)

### Mô hình
- [x] Tải `final_model.pt` + `kltn_model.py` của notebook v12 (`kaggle kernels output cngngquc/mainnotebook -p models/`) — `TwoTower` concat, input `[1536, 768]`, 20 nhãn, ngưỡng 0,75, `zscore`; load + predict thử trong `gpu-worker` được
- [x] Chép `kltn_model.py` vào `src/vicap/model/`, **giữ nguyên tên module** `kltn_model` (ruff bỏ qua file này); chỉ để `final_model.pt` trong `models/`
- [x] `vicap.model.predict.build_p1_inputs`: `text_emb` → 1.536 theo thứ tự `[caption+hashtag ; ASR]`, `image_emb` → trung bình các ảnh có thật (chia cho số ảnh có, không chia 8) + test
- [x] Kiểm tra `build_p1_inputs` trên cache Kaggle (`data/kaggle/mainnotebook/`, tải bằng `uvx --with six kaggle kernels output cngngquc/mainnotebook --file-pattern …`): 2.107 video, cosine với `V_frames` nhỏ nhất 0,9999998; `T_capasr` khớp tuyệt đối. Model cuối dùng `text_caphash_290eedc8` + `text_asr_chunked_PhoWhisper_medium_6ba74b77` (bản 2.107 video; hash khác là dataset cũ)
- [ ] Chạy `register_p1.py` (log với `code_paths=[.../kltn_model.py]`) → `kltn-p1` phiên bản 1, alias `@champion`, ngưỡng 0,75
- [ ] `vicap.model.predict.predict_batch`: gán nhãn theo `decide(..., at_least_one=True)` (PLAN mục 4.3)
- [ ] Kiểm tra `predict_batch` trên 100 video Gold khớp xác suất và nhãn của notebook

### Pipeline
- [x] `ingestion/`: crawler yt-dlp (theo kết quả Giai đoạn 0) → mp4 / ảnh slideshow + ảnh bìa + `info.json` (`itemStruct`) lên `raw/videos/{video_id}/` trước, rồi gửi message vào topic `tiktok.videos` (key `video_id`, metadata chính + key MinIO, PLAN mục 4.2); `vicap.pipelines.crawl.run_crawl` / `make crawl`; chạy thật 2 video + 1 slideshow (8 ảnh), message 0,7–1,1 KB, crawl lại không tải lại mp4
- [ ] Điền `configs/creators.txt`: 25–30 creator Đợt B (mới, không có trong dataset v5)
- [x] `bronze/`: consumer kafka-python đọc tới `end_offsets` (so `position`), đọc message bằng `VideoMessage.from_bytes`, đọc `info.json` từ `raw/` vào cột `raw_json`, khử trùng theo `(batch_id, video_id)` (giữ `crawl_ts` mới nhất), ghi đè `bronze.videos_raw` theo `batch_id`, rồi mới commit offset; message lỗi parse / sai schema và message `status="error"` → `ops.quarantine` — `vicap.pipelines.bronze.run_bronze` / `make bronze`; chạy thật 12 message (6 video × 2 lô) → 12 dòng, chạy lại 0 dòng; quarantine gộp theo `(stage, batch_id)` (`merge_quarantine_rows`), video đã vào Bronze thì bỏ khỏi quarantine
- [x] Log bằng Loguru (`vicap.utils.log.setup_logging`): màu theo level khi chạy ở terminal, chữ trơn trong Airflow; log `logging` chuẩn của thư viện được chuyển sang Loguru; image Airflow / gpu-worker build lại có `loguru`
- [x] Test integration không ghi vào bảng / bucket thật: fixture `iceberg_scratch` (namespace tạm), `kafka_topic` (topic tạm), test S3 tự xoá; đã dọn rác cũ trong `lakehouse/`
- [x] `silver/` (Spark): giữ bản crawl mới nhất mỗi video, caption / hashtag dựng như v5 (`vicap.silver.text`, so 186 video Giai đoạn 0), `silver_videos_suite` + `ops.quarantine`, `MERGE INTO silver.videos` theo `video_id` (chỉ cập nhật khi crawl sau) — `vicap.pipelines.silver.run_silver_clean` / `make silver`; chạy thật 12 dòng Bronze → 6 video, chạy lại 0 dòng, không snapshot mới; test Spark trong container: `make test-spark`
- [x] `features/media.py`: port 1-1 `extract_frame.py` + `extract_audio.py` (bản gốc ở `scripts/reference/`, ffmpeg của `imageio-ffmpeg`) → `derived/frames/`, `derived/audio/` + `silver.media` — `vicap.pipelines.media` / `make media`; so 181 video Giai đoạn 0: dHash 1.267/1.267 frame lệch 0 bit so với frame v5; chạy thật 6 video → 48 ảnh + 6 mp3
- [x] `features/asr.py` + `pipelines/asr.py`: PhoWhisper như cell 10 notebook → `silver.transcripts` (`transcript_status` ok / empty_or_music / missing / error) — `make asr`; chạy thật 6 video trên RTX 3050 → 6 `ok`; so 10 video v5 (mp4 Giai đoạn 0) với `transcript_full_v2.csv`: WER trung bình 8,5% (1–22%), lệch vài từ gần chỗ nối đoạn 30 s (GPU T4 → 3050 fp16, mp4 tải lại); chạy lại trên cùng máy cho kết quả giống hệt
- [x] Đo ảnh hưởng của chênh lệch ASR (60 video v5, mp4 Giai đoạn 0): cosine `T_asr` mới vs Kaggle median 0,998 (mean 0,953, min 0,56); P1 với `T_asr` mới: |Δxác suất| trung bình 0,0015 (max 0,167), **tập nhãn giống 60/60 video**
- [x] `features/text.py`: port PhoBERT của notebook (`word_segment`, `encode_texts`, `encode_chunked`); cosine với cache `T_cap` / `T_asr` = 1,0000 trên 60 video
- [x] `features/image.py`: port SigLIP2 của notebook (`siglip_encode`, `get_image_features` → `pooler_output`, fp32); 8 ảnh dựng lại từ mp4 + ảnh bìa Giai đoạn 0 (181 video, 1.448 ảnh): cosine từng ảnh với `V_frames_seq` min 0,9997 / mean 1,0000, P1 tập nhãn giống 181/181 (|Δxác suất| max 0,002)
- [x] `pipelines/embed.py`: PhoBERT → SigLIP2 (giải phóng VRAM giữa hai model) → `gold.features` (`replace_by_keys`, `gold_features_suite`, lỗi → `ops.quarantine`) — `make embed`; chạy thật 6 video, chạy lại bỏ qua 6; thử P1 trên 6 video: blv Anh Quân → Sports, Radio Việt Nam → Entertainment_Sing (xác suất 0,97–1,00)
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
- [ ] Task `discover`: tìm creator mới theo hashtag, lọc theo follower → topic `tiktok.creators` (key `username`; bỏ comment trong `create_topics.sh`) → Bronze writer ghi `bronze.creators_raw`
- [ ] DAG `recrawl`: crawl lại theo tần suất đăng
- [ ] Chọn cách gộp hồ sơ theo số video (dưới 10: trung bình xác suất; từ 10: đếm nhãn)
- [ ] Cờ đổi hướng nội dung (cosine hồ sơ mới so với cũ)
- [ ] `apps/backend/` FastAPI (`pip install .[api]`): `/creators` (lọc theo nhãn, số video), `/creators/{username}`, `/creators/{username}/history`, `/compare`; test bằng ClickHouse có dữ liệu v5
- [ ] `apps/frontend/` (Vite + React): màn Tìm creator, Hồ sơ creator, So sánh creator
- [ ] Thêm `backend`, `frontend` vào docker-compose profile `app` (`build: ./apps/backend`, `./apps/frontend`)

## Việc song song

- [ ] Thí nghiệm bổ sung trên OOF: luật loại trừ Others, phân tích theo loại video cho P1, hồ sơ P1 so với concat, creator 5–9 video
- [ ] Viết khóa luận: chương dữ liệu, phương pháp, kết quả
