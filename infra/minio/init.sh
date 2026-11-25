#!/bin/sh
# Tạo bucket + 2 user dịch vụ (PLAN mục 4.2, 9.4). Chạy lại được nhiều lần.
#   vicap-crawler : ghi raw/            (chỉ crawler)
#   vicap-pipeline: đọc raw/, ghi lakehouse/ + mlflow/
# Không bucket nào mở anonymous.
set -eu

# alias riêng (`local` là alias có sẵn của mc, trỏ localhost); alias set chỉ thành công khi MinIO đã lên.
i=0
until mc alias set vicap http://minio:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -gt 60 ] && { echo "MinIO chưa sẵn sàng sau 60 s" >&2; exit 1; }
  sleep 1
done
mc ready vicap

for b in raw lakehouse mlflow; do
  mc mb --ignore-existing "vicap/$b"
  mc anonymous set none "vicap/$b" >/dev/null
done

mc admin policy create vicap vicap-crawler /policies/crawler.json
mc admin policy create vicap vicap-pipeline /policies/pipeline.json

mc admin user add vicap "$MINIO_CRAWLER_ACCESS_KEY" "$MINIO_CRAWLER_SECRET_KEY"
mc admin user add vicap "$MINIO_PIPELINE_ACCESS_KEY" "$MINIO_PIPELINE_SECRET_KEY"

# attach báo lỗi nếu policy đã gắn sẵn → bỏ qua để chạy lại được
mc admin policy attach vicap vicap-crawler --user "$MINIO_CRAWLER_ACCESS_KEY" 2>/dev/null || true
mc admin policy attach vicap vicap-pipeline --user "$MINIO_PIPELINE_ACCESS_KEY" 2>/dev/null || true

mc admin user info vicap "$MINIO_CRAWLER_ACCESS_KEY" | grep -q vicap-crawler
mc admin user info vicap "$MINIO_PIPELINE_ACCESS_KEY" | grep -q vicap-pipeline
echo "minio-init xong: bucket raw, lakehouse, mlflow; user crawler + pipeline"
