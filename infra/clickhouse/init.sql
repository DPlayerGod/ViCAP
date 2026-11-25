-- Bảng phục vụ dashboard (Superset / FastAPI). Nạp từ 3 bảng gold Iceberg (src/vicap/iceberg/tables.py).
-- Chạy một lần khi volume ClickHouse còn trống (docker-entrypoint-initdb.d).
-- ReplacingMergeTree: nạp lại cùng khoá thì giữ bản có cột version lớn nhất; truy vấn dùng FINAL.

CREATE DATABASE IF NOT EXISTS vicap;

CREATE TABLE IF NOT EXISTS vicap.video_predictions
(
    video_id      String,
    username      String,
    probs         Array(Float32),   -- 20 xác suất, thứ tự nhãn như mô hình
    labels        Array(String),    -- decide(P, 0.75, at_least_one=True)
    threshold     Float32,
    model_name    String,
    model_version String,
    model_run_id  String,
    batch_id      String,
    predicted_at  DateTime64(6, 'UTC')
)
ENGINE = ReplacingMergeTree(predicted_at)
ORDER BY video_id;

CREATE TABLE IF NOT EXISTS vicap.creator_profiles
(
    username     String,
    n_videos     Int32,
    label_ratios Array(Float32),       -- 19 nhãn, bỏ Others
    main_label   Nullable(String),
    confidence   LowCardinality(String), -- low / medium / high
    model_run_id String,
    batch_id     String,
    updated_at   DateTime64(6, 'UTC')
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY username;

CREATE TABLE IF NOT EXISTS vicap.creator_profile_history
(
    snapshot_date Date,
    username      String,
    n_videos      Int32,
    label_ratios  Array(Float32),
    main_label    Nullable(String),
    confidence    LowCardinality(String),
    model_run_id  String,
    batch_id      String,
    updated_at    DateTime64(6, 'UTC')
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(snapshot_date)
ORDER BY (snapshot_date, username);
