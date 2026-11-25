# PLAN — ViCAP: Hệ thống phân loại nội dung và hồ sơ creator TikTok

Tài liệu này mô tả cách xây hệ thống xử lý theo lô cho khóa luận: thu thập video TikTok, phân loại nội dung từng video bằng mô hình P1 (Cấp 1), dựng và cập nhật hồ sơ creator (Cấp 2), và cho phép tra cứu, so sánh creator trên dashboard.

Danh sách việc cụ thể nằm trong `TODO.md`.

---

## 1. Mục tiêu

- Chạy được **đầu–cuối**: crawl → lưu trữ → xử lý → suy luận → hồ sơ → dashboard.
- **Cập nhật tăng dần**: mỗi lần chạy chỉ xử lý video mới; chạy lại không tạo bản ghi trùng (idempotent theo `video_id`).
- **Truy vết được**: mỗi dự đoán ghi kèm phiên bản mô hình (MLflow run id) và `batch_id`.
- **Tự mở rộng sang creator mới, nhỏ**: phát hiện creator qua hashtag theo nhãn, crawl lại creator cũ theo tần suất đăng.
- Chạy trên **một máy** bằng docker-compose; không dùng Kubernetes.

## 2. Kiến trúc

```
TikTok ──> Crawler (yt-dlp) ──┬─> MinIO raw/ (lớp 0: mp4, ảnh bìa, ảnh slideshow; ngoài Iceberg)
                              └─> Kafka (metadata + key raw/) ──> Bronze writer (Python + PyIceberg) ──> Iceberg
                                              Bronze ─(Spark; gpu-worker: PhoWhisper)─> Silver ─(gpu-worker: PhoBERT, SigLIP2)─> Gold
                                                                                                                              │
                                       MLflow (model registry: kltn-p1@champion) ──> predict P1 (gpu-worker) <───────────────┘
                                                                                         │
                                                                    ClickHouse (dự đoán, hồ sơ) ──> Superset
Airflow: điều phối mọi job theo lịch (crawl, Bronze, Spark, gpu-worker qua DockerOperator, nạp ClickHouse).
Kafka chạy liên tục, không do Airflow điều khiển.
raw/ chỉ crawler ghi; crawl lại cùng video thì ghi đè; Bronze giữ lịch sử metadata, raw/ chỉ giữ bản mới nhất.
Bronze trở đi chỉ lưu key trỏ vào raw/; gpu-worker đọc raw/ và ghi frame / audio sang lakehouse/derived/.
Vòng cập nhật: ClickHouse chọn creator cần crawl lại / creator mới phát hiện → Crawler.
```

| Tầng | Công cụ | Vai trò |
|---|---|---|
| Thu thập | yt-dlp (Python), chốt ở Giai đoạn 0; không dùng Apify | Lấy mp4, ảnh bìa (nguyên bytes), metadata JSON |
| Vận chuyển | Kafka (client: kafka-python) | Hàng đợi video mới giữa crawler và Bronze writer |
| Lưu trữ | MinIO (object storage) + Iceberg (table format) | Bucket `raw` (lớp 0, file gốc); các bảng Bronze / Silver / Gold |
| Xử lý | Spark (`silver_clean`, `profile`); PyArrow + PyIceberg (ghi Bronze, nạp ClickHouse, kiểm định) | Làm sạch, khử trùng theo `video_id`, tổng hợp hồ sơ |
| Đặc trưng | Container `gpu-worker` (CUDA + torch), Airflow gọi bằng DockerOperator: PhoWhisper, PhoBERT, SigLIP2 | Transcript (Silver), embedding text 2×768 + ảnh 8×768 (Gold) |
| Mô hình | MLflow + P1 (`final_model.pt`, ngưỡng 0,75); `predict` chạy trong `gpu-worker` | Dự đoán 20 nhãn (đa nhãn, mục 4.3); quản lý phiên bản |
| Phục vụ | ClickHouse + Superset | Truy vấn nhanh, dashboard phân tích nội bộ |
| Ứng dụng web | FastAPI (`apps/backend/`) + web (`apps/frontend/`) | Người dùng tìm creator theo nhóm nội dung, xem hồ sơ, so sánh creator |
| Điều phối | Airflow (LocalExecutor) | DAG theo ngày, thử lại khi lỗi |

## 3. Giai đoạn

### Giai đoạn 0 — Kiểm tra thu thập (làm trước tiên)
Trả lời câu hỏi: **yt-dlp có thay được Apify không?**
- Viết và chạy `scripts/test_collect.py` trên 5–10 creator, gồm cả creator có slideshow.
- Đo: tỉ lệ tải thành công, đủ trường metadata không, slideshow có tải được không, có bị chặn không, thời gian mỗi video.
- Đối chiếu metadata với dữ liệu Apify đang có (cùng `video_id`): caption, hashtag, thời lượng.
- **Tiêu chí qua:** ≥ 95% video thường tải được kèm đủ caption, hashtag, thời lượng, thumbnail. Nếu slideshow không tải được thì giữ Apify cho riêng slideshow.
- **Kết quả (đã xong): chốt yt-dlp cho toàn bộ.** Trên 186 video của dataset v5 (`scripts/phase0_compare.py`, không cookie): tải 186/186; caption (sau khi bỏ `#hashtag`) và hashtag khớp 186/186; ảnh bìa `cover` khớp MD5 186/186 (giống từng byte) và đúng kích thước; giờ trong tên file khớp UTC 186/186. 7 frame cắt theo chỉ số frame lệch dHash median 1 bit (89,7% frame ≤ 5 bit), sát hơn seek theo ms (median 4) → chốt cắt theo chỉ số frame (mục 4.1). Slideshow còn để mở (mục 4.1).

### Giai đoạn 1 — Bản chạy được đầu–cuối (MVP)
- docker-compose với: MinIO, Kafka, Spark, Airflow (+ Postgres), MLflow, ClickHouse, Superset; image `gpu-worker` riêng.
- Nạp dataset v5 (2.107 video, embedding có sẵn) vào Bronze / Silver / Gold để có dữ liệu ngay, không phải chạy lại GPU.
- Đăng ký P1 vào MLflow (`register_p1.py`, alias `@champion`).
- DAG `daily_pipeline` chạy đủ các bước trên **dữ liệu Đợt B** (creator mới, 10 video gần nhất, không chọn lọc).
- Hồ sơ creator ở MVP: trung bình xác suất 19 nhãn, giống `kltn_model.profile` (mục 4.3).
- Dashboard Superset: lọc creator theo nhóm nội dung, xem hồ sơ và lịch sử hồ sơ.

### Giai đoạn 2 — Mở rộng (nếu còn thời gian)
- Phát hiện creator mới tự động theo hashtag của từng nhãn, lọc creator nhỏ theo follower.
- Crawl lại theo tần suất đăng (đăng nhiều: mỗi tuần; đăng ít: mỗi tháng).
- Cơ chế chọn cách gộp hồ sơ theo số video (dưới 10 video: trung bình xác suất; từ 10 video: đếm nhãn quyết định).
- Cờ "đổi hướng nội dung" khi cosine giữa hồ sơ mới và cũ thấp hơn ngưỡng.
- Ứng dụng web: `apps/backend/` FastAPI đọc ClickHouse (`GET /creators?label=&min_videos=`, `GET /creators/{username}`, `GET /creators/{username}/history`, `GET /compare?u=a&u=b`) và `apps/frontend/` gồm 3 màn: tìm creator theo nhãn, hồ sơ creator (biểu đồ tỉ lệ nhãn + video gần nhất), so sánh 2–3 creator. Superset vẫn giữ cho phân tích nội bộ.

## 4. Mô hình dữ liệu

| Bảng (Iceberg) | Nội dung chính |
|---|---|
| `bronze.videos_raw` | `video_id`, `username`, `crawl_ts`, `batch_id`, `media_type` (video / slideshow, mục 4.1), metadata JSON gốc, key mp4 / thumbnail trên MinIO (theo `video_id`, mục 4.2) |
| `bronze.creators_raw` | `username`, follower, nguồn phát hiện (hashtag nào), thời điểm phát hiện |
| `silver.videos` | Video đã làm sạch, khử trùng; `media_type` (video / slideshow), caption + hashtag đã chuẩn hoá |
| `silver.transcripts` | Transcript PhoWhisper (`transcript_raw`), cờ có / không có transcript |
| `gold.features` | `text_emb` float32 2×768, `image_emb` float32 8×768 (mục 4.1), tên + phiên bản encoder |
| `gold.video_predictions` | 20 xác suất, `labels` (mục 4.3), ngưỡng, `model_run_id`, `batch_id` |
| `gold.creator_profiles` | Hồ sơ hiện tại: số video, tỉ lệ 19 nhãn (bỏ Others), nhóm chính, mức tin cậy (mục 4.3) |
| `gold.creator_profile_history` | Ảnh chụp hồ sơ theo ngày |
| `ops.quarantine` | Dòng bị cách ly ở mọi bước: `video_id`, `username`, `stage`, `batch_id`, `processing_date`, `errors`, `severity`, `raw_ref`, `resolved`, `created_at` (mục 9.3) |

ClickHouse giữ bản sao của `gold.video_predictions`, `gold.creator_profiles`, `gold.creator_profile_history` để dashboard truy vấn.

### 4.1. Đặc trưng (`gold.features`)

`gold.features` có hai cột, đều lưu dạng float32:

- `text_emb`: 2 × 768.
  - `[0]`: PhoBERT của caption + hashtag. Ghép theo `caption + " " + " ".join(hashtag)`, tách từ bằng pyvi, cắt ở 256 token, rồi mean-pool theo mask.
  - `[1]`: PhoBERT của transcript thô (`transcript_raw`, không dùng bản đã làm sạch). Transcript được chia thành các đoạn ≤ 254 token, mỗi đoạn mean-pool, rồi lấy trung bình các đoạn. Video không có transcript thì vector này bằng 0.
- `image_emb`: 8 × 768, là `pooler_output` của SigLIP2 `siglip2-base-patch16-224`, chưa chuẩn hoá L2.
  - Thứ tự cố định: ảnh bìa `_0`, rồi frame `_1…_7` lấy tại 12,5%–87,5% thời lượng.
  - **8 ảnh trong `derived/frames/` phải dựng giống hệt dataset `tiktok-frames-v1` lúc train**, tức port nguyên `03_src/scripts/extract_frame.py` (bản tạo `{video_id}_{k}.jpg`; `extract_frames.py` với `_f{k}`, 448px là bản cũ, không dùng):
    - `_0` lấy từ `raw/videos/{id}/cover.*` (ảnh bìa lưu nguyên bytes trong `raw`, đuôi theo magic bytes; dataset v5 có 410/2.107 ảnh bìa PNG). Không có ảnh bìa thì dùng frame tại 0 s.
    - `_k` (k = 1…7) cắt **theo thời gian bằng ffmpeg**: `ffmpeg -ss {dur·k/8} -i video.mp4 -frames:v 1` (seek trước `-i`); `dur` đọc từ header container (`ffmpeg -i`, không dùng `duration` của metadata), chặn ở `dur − 0,1 s`. Không ra frame thì thử lại tại 0,9 / 0,75 / 0,5 / 0,25 / 0 lần mốc đó.
    - Cả 8 ảnh qua cùng một bước: `convert("RGB")` → `PIL.Image.thumbnail((640, 640))` (cạnh dài ≤ 640, giữ tỉ lệ, chỉ thu nhỏ) → JPEG `quality=90` (subsampling 4:2:0 mặc định của Pillow). Vì vậy `_0` luôn là `.jpg`, kể cả khi ảnh bìa gốc là PNG.
    - Đã đo trên 48 ảnh của 6 video trong dataset cũ: cạnh dài đều 640 (vd 360×640; một `_0` là 447×640), JPEG chất lượng ≈ 90, 4:2:0, khớp script.
    - Giai đoạn 0 chưa có script gốc nên chỉ so hai cách dựng lại với frame v5: theo chỉ số frame OpenCV (`CAP_PROP_POS_FRAMES`, dHash lệch median 1 bit) và seek theo ms từ `duration` của metadata (median 4 bit). Cách của `extract_frame.py` (ffmpeg `-ss` theo `dur` của container) chưa đo; cần chạy `phase0_compare.py` thêm cách này, kỳ vọng lệch ≈ 0 bit. Nếu đúng thì pipeline port nguyên `extract_frame.py`.
    - SigLIP2 còn resize về 224 nên sai lệch ở bước này nhỏ, nhưng làm đúng để Đợt B so được với báo cáo và để `derived/` không chứa ảnh 2160×3840.
  - Nếu số ảnh khác 8 thì lấy mẫu đều về 8 ảnh bằng linspace. Nếu không có ảnh nào thì vector bằng 0.
- **Audio cho PhoWhisper** (`derived/audio/{video_id}.mp3`), giống dataset `tiktok-audio-v1` (`03_src/scripts/extract_audio.py`): `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 -b:a 64k {video_id}.mp3` → mp3 (MPEG-2 Layer III) 16 kHz mono 64 kbps CBR (đo trên 6 file cũ; docstring script ghi "AAC" nhưng file thực là mp3). Audio < 1 s coi là không có transcript (`too_short`). Script cũ chỉ tách audio từ `.mp4`, slideshow không có audio.

**Slideshow — còn để mở.** yt-dlp chỉ trả audio + ảnh bìa; ảnh của slideshow lấy được từ `imagePost` trong itemStruct (hàm nội bộ `TikTokIE._extract_web_data_and_status`, cần ghim phiên bản yt-dlp). Dataset v5 vẫn có `_1…_7` cho slideshow nhưng chưa rõ cách dựng, nên chưa chốt cách lấy 8 ảnh. Trong lúc chờ: gắn cờ `media_type` (`video` / `slideshow`) ngay từ Bronze, theo `vcodec` / có `imagePost`, để tách riêng khi đánh giá và khi tính embedding.

`model/predict.py` dựng đầu vào cho P1:

- Text: nối `text_emb` thành 1.536 chiều (tương ứng `T_capasr` trong notebook).
- Hình: lấy trung bình 8 vector của `image_emb` thành 768 chiều (tương ứng `V_frames`).

Kiểm tra trong `quality/suites.py` (mức "cách ly dòng", mục 9.3):

- `text_emb` đúng 2 × 768, `image_emb` đúng 8 × 768, không có NaN.
- `text_emb[0]` và cả 8 vector ảnh đều khác 0.
- `text_emb[1]` được phép bằng 0 vì có video không có lời nói.

### 4.2. Quy ước đặt tên file

Các file trong dự án đang dùng ba kiểu tên.

**1. Dữ liệu gốc: `{username}__{YYYYMMDD_HHMMSS}__{video_id}`**

Các phần cách nhau bằng hai dấu gạch dưới. Phần thời gian là lúc video được đăng, theo **UTC** (`createTime` của TikTok; đã kiểm tra trên cả hai ví dụ dưới đây ở Giai đoạn 0).

| Loại | Ví dụ |
|---|---|
| Video | `02_data/raw/videos/1vongthread__20260709_094500__7660442169215487252.mp4` |
| Slideshow (thư mục) | `02_data/raw/videos/_vetthuong_09__20261008_153532__7694321051815185682/` |
| Ảnh trong slideshow | `…7694321051815185682__img_001.jpg`, `__img_002.jpg`, … |
| Thumbnail | `02_data/raw/thumbnails/1vongthread__20260709_094500__7660442169215487252.jpg` |
| Thumbnail trên Kaggle | `tiktok-multilabel-content/thumbnails/` dùng đúng tên này |

**2. Dữ liệu đã xử lý: chỉ dùng `{video_id}`**

| Loại | Ví dụ |
|---|---|
| Frame | `tiktok-frames-v1/video_frames/6917211883481074946_0.jpg` … `_7.jpg` (`_0` là thumbnail, `_1` đến `_7` là frame) |
| Audio | `tiktok-audio-v1/audio/6917211883481074946.mp3` |

**Cách tách tên.** Username có thể chứa dấu `_` (ví dụ `_vetthuong_09`), nên không tách từ bên trái. Bỏ hậu tố `__img_NNN` (nếu có) và phần mở rộng trước, rồi `rsplit("__", 2)` → `(username, posted_at, video_id)`. Code nằm ở `vicap.storage.paths`.

**`legacy_stem` — chỉ để ghép với dữ liệu v5.** Kiểu tên 1 ở trên gọi là `legacy_stem`. Hệ thống mới **không** dùng nó làm key lưu trữ; nó chỉ dùng để ghép file / dòng của dataset v5 (Kaggle, `02_data/raw/`) với `video_id`. Lưu ý khi dựng lại `legacy_stem` từ metadata: dataset v5 **bỏ dấu `.`** trong username (`harry.nista` → `harrynista__20260131_060926__7601403892349275399`; gặp ở 5 creator trong Giai đoạn 0), nên ghép theo `video_id` (phần cuối sau `rsplit`) chứ không theo username.

**Bố cục trên MinIO — key theo `video_id`.** Media dùng key cố định theo `video_id` (không phụ thuộc username, tránh lỗi khi creator đổi tên); crawl lại cùng video thì ghi đè đúng file cũ, không sinh bản trùng. Bronze giữ lịch sử metadata của mọi lần crawl, còn `raw/` chỉ giữ bản mới nhất: dòng Bronze cũ trỏ vào file mới, nên nếu creator đổi ảnh bìa thì lịch sử không khớp ảnh. Chấp nhận được cho khóa luận; muốn giữ mọi phiên bản thì bật versioning cho bucket `raw` trên MinIO (không phải sửa code). Key dựng bằng `vicap.storage.paths`, đọc / ghi bằng `vicap.storage.objects` (`put_bytes`, `get_bytes`, `exists`).

- Bucket `raw` — **lớp 0**, nằm ngoài Iceberg, **chỉ crawler ghi**. Mỗi video một thư mục, `raw_video(video_id, name)`:
  - `videos/{video_id}/video.mp4` (video thường)
  - `videos/{video_id}/img_{NNN}.{jpg|png}` (slideshow, đánh số từ 1) — **chưa có, còn để mở**: yt-dlp chưa tải được ảnh slideshow (Giai đoạn 0), xem "Slideshow — còn để mở" ở mục 4.1. Key đã định sẵn để crawler dùng khi có cách lấy ảnh.
  - `videos/{video_id}/cover.{jpg|png}` (ảnh bìa, nguyên bytes, mục 4.1)
- Bucket `lakehouse`:
  - `derived/frames/{video_id}_{k}.jpg` (`frame(video_id, k)`; cả 8 ảnh JPEG 640px, mục 4.1), `derived/audio/{video_id}.mp3` (16 kHz mono 64 kbps): gpu-worker sinh từ `raw`
  - `quarantine/{domain}/processing_date=…/batch_id=…/` (`quarantine(domain, date, batch_id)`): **chỉ lô bị dừng** (`data.parquet` + `errors.json`); dòng bị cách ly nằm ở bảng `ops.quarantine` (mục 9.3)
  - `warehouse/`: bảng Iceberg

Không có tầng staging Parquet như HOSE: metadata đi qua Kafka vào thẳng `bronze.videos_raw` (cột `raw_json` + key `media_key`, `thumbnail_key` trỏ vào `raw`).

### 4.3. Gán nhãn và hồ sơ

- P1 trả 20 xác suất (sigmoid, đa nhãn). `labels` = mọi nhãn có p ≥ 0,75; nếu không nhãn nào đạt thì lấy nhãn có xác suất cao nhất (argmax). Đây là `decide(P, th, at_least_one=True)` trong notebook, khớp với báo cáo ("luôn giữ ít nhất top-1"). Others chỉ xuất hiện khi chính nó có xác suất cao nhất.
- `decide_rule` (`THRESH_MODE="rule"`, rỗng thì gán Others) mới để sẵn, chưa dùng và chưa đánh giá; pipeline **không** dùng, để F1 trên Đợt B so được với số trong báo cáo.
- Hồ sơ ở MVP: tỉ lệ 19 nhãn (bỏ Others) = trung bình xác suất trên các video của creator, giống `kltn_model.profile`.
- Mức tin cậy theo số video: **thấp** khi < 5 video, **trung bình** khi 5–9, **cao** khi ≥ 10.

## 5. DAG `daily_pipeline`

```
discover ──> crawl ──> bronze ──> silver_clean ──> asr ──> embed ──> predict ──> profile ──> load_clickhouse ──> quality_check
```

- **`batch_id`** = `YYYYMMDD-HHMMSS` lấy từ logical date của DAG run. Khi retry, task dùng lại cùng id nên vẫn idempotent.
- **Ghi theo khoá:**
  - Bronze ghi đè theo `batch_id`, giữ lịch sử các lần crawl lại.
  - Silver / Gold ghi theo `video_id` (giữ bản crawl mới nhất). Spark dùng `MERGE INTO` (cần Iceberg SQL extensions, mục 9.4).
  - Với PyIceberg, bảng có cột mảng (`gold.features`, `gold.video_predictions`) dùng `overwrite(df, overwrite_filter=In("video_id", ids))`: xoá rồi ghi trong cùng một commit, kết quả giống upsert. Không dùng `table.upsert` cho các bảng này, vì hàm đó so từng cột không phải khoá để tìm dòng đổi, gặp mảng 8×768 thì dễ lỗi hoặc rất chậm. `upsert` chỉ dùng cho bảng không có cột mảng.
- **Bronze đọc Kafka** (kafka-python, consumer group `bronze-writer`, tắt auto commit):
  - Đầu task lấy `end_offsets(assignment)` làm mốc.
  - Poll tới khi mọi partition có `consumer.position(tp) >= mốc`.
  - Ghi Iceberg xong mới commit offset. Lỗi giữa chừng thì không commit; lần chạy lại đọc lại từ offset cũ.
- GPU worker (container `gpu-worker`, mục 9.4) nạp **lần lượt** PhoWhisper → PhoBERT → SigLIP2, giải phóng bộ nhớ sau mỗi bước.
- `predict` nạp `models:/kltn-p1@champion`, gán nhãn theo mục 4.3; đổi mô hình chỉ cần chuyển alias trong MLflow.
- Mỗi bước kiểm định theo mục 9.3: dòng lỗi bị cách ly, phần còn lại chạy tiếp; lỗi cấp lô thì dừng.
- `quality_check`: số video mỗi lô, tỉ lệ thiếu transcript / embedding, số dòng bị cách ly, phân bố nhãn so với lô trước.

## 6. Tài nguyên

| Thành phần | RAM ước lượng |
|---|---|
| Spark (local) | 2–4 GB |
| Kafka (KRaft, 1 broker) | ~1 GB |
| ClickHouse | 1–2 GB |
| Airflow + Postgres | ~1,5 GB |
| MinIO, MLflow, Superset | ~1,5 GB |
| GPU worker | VRAM 4 GB, RAM ~3 GB |

Tổng khoảng 10–13 GB RAM khi chạy cùng lúc. Nếu máy không đủ: dùng docker-compose profile để chỉ bật dịch vụ cần cho bước đang làm, hoặc chạy demo trên VPS / máy khác.

## 7. Đánh giá hệ thống

- **Đợt B:** F1 Cấp 1 và độ đúng hồ sơ Cấp 2 trên creator mới, video không chọn lọc, dùng ngưỡng 0,75 và luật gán nhãn ở mục 4.3.
- **Đợt B2:** crawl lại cùng creator → pipeline chỉ xử lý video mới, không trùng; hồ sơ cập nhật hợp lý.
- **Vận hành:** thời gian mỗi lô, số video / giờ trên GPU 4 GB, tỉ lệ task phải chạy lại.

## 8. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| TikTok chặn / yt-dlp hỏng khi TikTok đổi web | Cập nhật yt-dlp, giãn tốc độ crawl, dùng cookie; Apify làm phương án dự phòng |
| Slideshow: yt-dlp không trả ảnh, cách dựng 8 ảnh chưa chốt | Lấy ảnh qua `imagePost` (hàm nội bộ, ghim phiên bản yt-dlp); gắn cờ `media_type` để tách riêng slideshow khi tính embedding và đánh giá (mục 4.1) |
| Thiếu RAM khi chạy đủ dịch vụ | Docker-compose profile; VPS cho demo |
| Embedding lệch phiên bản encoder | Cố định tên + phiên bản encoder trong config; ghi vào `gold.features` |
| Ngưỡng 0,75 chưa tối ưu trên dữ liệu thật | Kiểm chứng trên Đợt B; đăng ký phiên bản mới trong MLflow nếu cần |
| Airflow mount `/var/run/docker.sock` để chạy DockerOperator: tiến trình trong Airflow có quyền gần như root trên host | Chấp nhận vì chỉ chạy local trên một máy; không mở cổng Airflow ra Internet |
| Một video lỗi (thiếu ảnh, slideshow hỏng) chặn cả lô | Kiểm định hai cấp (mục 9.3): chỉ cách ly dòng lỗi |

## 9. Codebase nền: kế thừa từ `hose-stock-lakehouse`

Hệ thống dựng trên repo đồ án thực tập [`DPlayerGod/hose-stock-lakehouse`](https://github.com/DPlayerGod/hose-stock-lakehouse). Hai hệ thống dùng chung hạ tầng (MinIO, Iceberg REST catalog trên Postgres, Kafka KRaft, ClickHouse, Airflow), nên không phải dựng lại từ đầu.

### 9.1. Giữ lại

| Phần của HOSE | Dùng trong ViCAP |
|---|---|
| `docker-compose.yml` chia profile, có `mem_limit`, các service `*-init` (bucket, schema) | Giữ cách làm; đổi profile thành `core / stream / pipeline / ml / serving / app` |
| `config.py` (dataclass đọc `.env`) | Giữ; thêm `KafkaConfig`, `MlflowConfig`, `ModelConfig` (ngưỡng, `at_least_one`, encoder) |
| `iceberg/` (catalog, reader, writer, tables) | Giữ; viết lại schema trong `tables.py` cho bảng video; thêm `replace_by_keys` (mục 5) |
| `quality/` (6 chiều, `run_suite`, quarantine) | Giữ thiết kế config + hàm check, `result.py`; viết lại `checks.py` trên PyArrow + NumPy thay cho Polars (bỏ `WithinDailyBand` của chứng khoán); thêm check vector và `split_valid`; viết `suites.py` mới cho video (mục 9.3) |
| `staging/` (`writer.py`: `StagingPathBuilder`, đọc / ghi Parquet; `quarantine.py`) | Đổi thành `storage/`: `paths.py` (hàm dựng key thay `StagingPathBuilder`: `raw_video`, `frame`, `quarantine`, cùng quy ước tên mục 4.2), `quarantine.py` (giữ nguyên, chỉ đổi đường dẫn; chỉ còn lỗi cấp lô, dòng lỗi chuyển sang bảng `ops.quarantine` trong `quality/quarantine.py`), `objects.py` (`put_bytes` / `get_bytes` / `exists`, crawler và gpu-worker dùng chung). Bỏ `write_staging_parquet` / `read_staging_parquet` |
| `utils/dates.py` | Giữ (`storage/paths.py` dùng `format_date`) |
| `clickhouse/` (client, loader), `streaming/clickhouse/init*.sql` | Giữ client; viết DDL và loader mới cho 3 bảng phục vụ |
| `streaming/producer/` | Thành producer đẩy metadata video vào Kafka |
| `tests/` (pytest, marker `integration`) | Giữ cách tổ chức |

Bỏ: `dnse_sdk/`, `streaming/alerts/`, `streamlit_app/` (dùng Superset), các DAG chứng khoán.

### 9.2. Cải thiện khi chuyển sang

1. **DAG mỏng.** Trong HOSE, `pipelines/` là các hàm chạy trọn chuỗi trong một lần gọi (`run_daily_ohlcv_pipeline`: extract → … → ClickHouse), còn `dag_daily_ohlcv.py` viết lại toàn bộ logic trong từng task. Ở ViCAP, logic từng bước nằm trong `bronze/`, `silver/`, `features/`, `model/`, `gold/`; `vicap.pipelines` có `run_<bước>(batch_id)` cho từng bước (đọc config, gọi hàm của bước, kiểm định, ghi) và `run_daily(batch_id)` nối tất cả để chạy không cần Airflow. Mỗi task trong `dags/` chỉ gọi một `run_<bước>`, không chép lại logic. Mọi bước test được bằng pytest.
2. **Silver đọc từ Bronze.** HOSE `transform_silver` đọc lại staging rồi dựng lại Bronze. Ở ViCAP, mỗi tầng chỉ đọc từ bảng của tầng ngay trước đó.
3. **Ghi theo khoá, không chỉ theo ngày.** HOSE ghi đè theo ngày. Video có thể bị crawl lại ở ngày khác, nên Silver / Gold khử trùng theo `video_id` (giữ bản crawl mới nhất), còn Bronze ghi đè theo `batch_id` (chi tiết ở mục 5).
4. **Đóng gói chuẩn.** HOSE yêu cầu Python `>=3.13` nhưng image Airflow là 3.12, và dependency nằm ở hai `requirements.txt` trùng nhau. Ở ViCAP: `requires-python >=3.11`; khai báo dependency trong `pyproject.toml` theo nhóm `[airflow]`, `[spark]`, `[gpu]`, `[api]`, `[dev]`; image Airflow cài package bằng `pip install .[airflow,spark]`.
5. **Bắt lỗi đúng.** `ensure_table` bắt `Exception` chung, nên lỗi mạng tới catalog cũng dẫn tới tạo bảng mới. Đổi sang bắt `NoSuchTableError`, và `NamespaceAlreadyExistsError` khi tạo namespace. `try_read_table` trong `reader.py` cũng bắt `Exception`, sửa tương tự.
6. **Cấu hình an toàn hơn.**
   - Bỏ mật khẩu mặc định (`admin123`) trong `config.py`: thiếu biến môi trường thì báo lỗi khi khởi tạo config (`_require_env()` trong `default_factory`).
   - Mọi tham số `config: X = X()` dùng làm default trong hàm đổi thành `config: X | None = None`; nếu không, chỉ cần import module là đã lỗi khi thiếu biến môi trường.
   - Ghim phiên bản mọi image (không dùng `latest`).
   - Bỏ `mc anonymous set public` (HOSE mở bucket ra public); Airflow admin không mặc định `admin/admin`; `AIRFLOW__WEBSERVER__EXPOSE_CONFIG=false`.
7. **Chạy trên Linux qua SSH.** Thay các script `.ps1` bằng `Makefile` (`make up PROFILE=core`, `make test`, `make dag-trigger`).
8. **Dọn repo.** Không commit báo cáo PDF và file dữ liệu ở thư mục gốc; sửa lỗi gõ `.DS_Stor` trong `.gitignore`; `models/` và `data/` để trong `.gitignore`.

### 9.3. Kiểm định chất lượng dữ liệu cho video

| Chiều | Mức | Kiểm tra | Bảng |
|---|---|---|---|
| Đầy đủ | cách ly dòng | Có `video_id`, `username`, file media, ảnh bìa | Bronze |
| Đầy đủ | cách ly dòng | Mọi video ở Silver đều có embedding | Gold `features` |
| Duy nhất | dừng lô | `video_id` không trùng | Silver, Gold |
| Hợp lệ | dừng lô | Đúng schema; cột embedding đúng kiểu cho cả cột | Silver, Gold |
| Hợp lệ | cách ly dòng | Thời lượng > 0; `text_emb` đúng 2×768, `image_emb` đúng 8×768, không NaN; `text_emb[0]` và 8 vector ảnh khác 0 (`text_emb[1]` được phép bằng 0); xác suất trong [0, 1] | Silver, Gold |
| Nhất quán | cách ly dòng | Mỗi dự đoán có video ở `features`; mỗi video thuộc một creator trong `creators` | Gold |
| Kịp thời | dừng lô | Lô không rỗng | Bronze |
| Kịp thời | cảnh báo | Ngày đăng không sau thời điểm crawl | Bronze |
| Hợp lý | cảnh báo | Tỉ lệ thiếu transcript và phân bố nhãn dự đoán không lệch nhiều so với lô trước | Silver, Gold |

Ba mức:

- **Dừng lô** (sai schema, cả cột sai shape, lô rỗng, trùng `video_id`): cả lô được ghi vào `quarantine/` trên MinIO (`data.parquet` + `errors.json`, `vicap.storage.quarantine`) để xem lại lô hỏng (`run_suite` + `quarantine_and_raise`, như HOSE), rồi DAG dừng.
- **Cách ly dòng** (một video thiếu ảnh, embedding bằng 0, …): `split_valid(df) -> (ok, bad)` tách dòng lỗi; `bad` được ghi vào bảng Iceberg **`ops.quarantine`** (`vicap.quality.quarantine.write_quarantine_rows`), `ok` chạy tiếp. Một slideshow lỗi không chặn cả ngày.
  - Cột: `video_id`, `username`, `stage`, `batch_id`, `processing_date` (partition), `errors` (nhãn check, vd `non_zero_vector:text_emb[0]`), `severity` (`error` / `warn`), `raw_ref` (`s3://raw/videos/{video_id}/`), `resolved`, `created_at`.
  - Ghi đè theo `(stage, batch_id)`: retry không sinh dòng trùng; lần chạy lại hết lỗi thì dòng cũ của lô đó bị xoá.
  - Dạng bảng thay cho file Parquet để đếm, lọc, join, đưa lên Superset và chạy lại dễ hơn.
- **Cảnh báo:** chỉ ghi vào kết quả `quality_check`.

Chiều "Hợp lý" ứng với `Dimension.ACCURACY` trong code.

### 9.4. Engine xử lý

- **Iceberg catalog:** REST (`tabulario/iceberg-rest`, JDBC trên Postgres, như HOSE), dùng chung cho PyIceberg và Spark.
- **Spark (local mode, chạy trong container Airflow):** hai job tổng hợp là `silver_clean` và `profile`. Không dựng Spark cluster riêng để tiết kiệm RAM.
  - HOSE không có Spark, nên image Airflow cài thêm OpenJDK 17, `pyspark` 3.5.x và hai jar `iceberg-spark-runtime-3.5_2.12`, `iceberg-aws-bundle`.
  - SparkSession bắt buộc có `spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions`; thiếu dòng này thì `MERGE INTO` báo lỗi cú pháp.
- **PyArrow + PyIceberg** (tiến trình Python trong Airflow): ghi Bronze, nạp ClickHouse. HOSE dùng Polars; ViCAP bỏ Polars vì PyIceberg đọc / ghi Arrow sẵn, kết quả Spark đưa sang Arrow được (Spark 3.5 chưa có `DataFrame.toArrow()` public — từ 4.0 mới có — nên dùng `toPandas()` với `spark.sql.execution.arrow.pyspark.enabled=true` rồi `pa.Table.from_pandas`; mỗi lô vài trăm video nên gom về driver không đáng kể), và check embedding làm bằng NumPy trên cả cột.
- **`gpu-worker`** (container riêng): PhoWhisper, PhoBERT, SigLIP2 và `predict` P1. Image này đã có torch và mlflow, nên image Airflow không phải cài torch.
  - NVIDIA Container Toolkit cài trên **host**. Dockerfile chỉ cần base image CUDA và torch (cộng transformers, pyvi, mlflow, ffmpeg).
  - Airflow gọi bằng DockerOperator với `device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])]` và `network_mode="lakehouse_network"` (cùng network compose), để container gọi được MinIO, Iceberg REST và MLflow.
  - Airflow mount `/var/run/docker.sock`. Đây là đánh đổi bảo mật (mục 8), chấp nhận được vì chỉ chạy local.
- **Dịch vụ phụ:** Postgres tạo thêm DB `mlflow` (backend store của MLflow, artifact ở bucket `mlflow`) và `superset`; image Superset cài `clickhouse-connect`.

### 9.5. Cấu trúc repo

```
ViCAP/
├── src/vicap/                 # thư viện dùng chung (src layout, PyPA)
│   ├── config.py
│   ├── utils/                 # dates                                  (từ HOSE)
│   ├── iceberg/               # catalog, reader, writer, tables        (từ HOSE)
│   ├── quality/               # checks (PyArrow + NumPy), result (từ HOSE) + suites video, split_valid, quarantine (ops.quarantine)
│   ├── storage/               # paths (key MinIO + tên file), objects, quarantine cấp lô (từ HOSE)
│   ├── clickhouse/            # client (từ HOSE), loader
│   ├── ingestion/             # crawler yt-dlp, producer Kafka
│   ├── bronze/                # Kafka → bronze.videos_raw
│   ├── silver/                # silver_clean (Spark), transcripts
│   ├── features/              # asr.py, embed.py (GPU), trích frame
│   ├── model/                 # kltn_model.py, register_p1.py, predict.py
│   ├── gold/                  # profile (Spark)
│   └── pipelines/             # run_<bước>(batch_id), DAG gọi vào đây
├── dags/                      # chỉ định nghĩa DAG (quy ước Astronomer)
│   ├── daily_pipeline.py
│   ├── recrawl.py
│   └── .airflowignore
├── apps/                      # ứng dụng chạy được (quy ước monorepo)
│   ├── backend/               # FastAPI: app/{main.py,routers,schemas,services}, Dockerfile, tests/
│   └── frontend/              # Vite + React: src/{pages,components,api}, Dockerfile, package.json
├── infra/                     # Dockerfile + cấu hình từng service
│   ├── airflow/Dockerfile     # JDK 17 + jar Iceberg Spark + pip install .[airflow,spark]
│   ├── gpu_worker/Dockerfile  # CUDA + torch, pip install .[gpu]
│   ├── clickhouse/init.sql
│   ├── kafka/create_topics.sh
│   └── superset/              # Dockerfile (+clickhouse-connect), export dashboard
├── configs/                   # hashtags.yaml, lịch crawl lại (dữ liệu cấu hình, không phải code)
├── scripts/                   # việc làm một lần: test_collect, phase0_compare, probe_cover
├── tests/                     # unit + integration cho src/vicap
├── docker-compose.yml         # ở gốc để chạy `docker compose up` từ gốc
├── pyproject.toml  uv.lock  Makefile  .env.example  .python-version
└── PLAN.md  TODO.md  README.md  LICENSE
```

`models/`, `data/` nằm trong `.gitignore`; notebook Kaggle chỉ để tham khảo, không đưa vào repo.

Ba quyết định đi kèm:

- **`docker-compose.yml` ở gốc**, không đưa vào `infra/`: `docker compose up` chạy ngay từ gốc, và `build: ./infra/airflow` đọc rõ ràng.
- **Backend dùng lại `vicap`:** nhóm dependency `[api]` (fastapi, uvicorn) trong `pyproject.toml` gốc; Dockerfile backend chạy `pip install .[api]`. Nếu sau này backend cần dependency riêng thì chuyển sang uv workspace (`apps/backend/pyproject.toml` với `vicap = { workspace = true }`); với khóa luận chưa cần.
- **`apps/frontend/`** có công cụ riêng (`package.json`, `node_modules`), độc lập với Python; `apps/frontend/node_modules/` nằm trong `.gitignore`.

**Lưu ý về `model/kltn_model.py`:** giữ nguyên tên module `kltn_model`. Mô hình trong MLflow được log với `code_paths=[kltn_model.py]`, và lúc load sẽ chạy `import kltn_model`. Đổi tên file hoặc đổi import thì `models:/kltn-p1@champion` không load được, trừ khi đăng ký lại.
