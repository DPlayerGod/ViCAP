#!/bin/sh
# Tạo topic Kafka (PLAN mục 5). auto.create.topics tắt nên phải tạo trước.
# Kafka chỉ nằm ở chỗ crawler giao dữ liệu cho pipeline; từ Bronze trở đi các bước truyền qua bảng Iceberg.
# Chạy lại nhiều lần được: topic đã có thì tăng partition (Kafka không giảm được) và đặt lại retention.
set -eu

BOOTSTRAP="${KAFKA_BOOTSTRAP:-kafka:9092}"
PARTITIONS="${KAFKA_PARTITIONS:-3}"
RETENTION_MS="${KAFKA_RETENTION_MS:-604800000}" # 7 ngày: Bronze mới là nơi lưu lâu dài
BIN=/opt/kafka/bin

i=0
until "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --list >/dev/null 2>&1; do
  i=$((i + 1))
  [ "$i" -gt 60 ] && { echo "Kafka chưa sẵn sàng sau 60 lần thử" >&2; exit 1; }
  sleep 2
done

ensure_topic() {
  topic="$1"
  "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --create --if-not-exists \
    --topic "$topic" --partitions "$PARTITIONS" --replication-factor 1 \
    --config retention.ms="$RETENTION_MS"

  current=$("$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --describe --topic "$topic" \
    | sed -n 's/.*PartitionCount: *\([0-9]*\).*/\1/p' | head -n 1)
  if [ "$current" -lt "$PARTITIONS" ]; then
    "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --alter --topic "$topic" --partitions "$PARTITIONS"
  fi
  "$BIN/kafka-configs.sh" --bootstrap-server "$BOOTSTRAP" --alter \
    --entity-type topics --entity-name "$topic" --add-config retention.ms="$RETENTION_MS"

  "$BIN/kafka-topics.sh" --bootstrap-server "$BOOTSTRAP" --describe --topic "$topic"
}

# Giai đoạn 1 — crawler → Bronze writer, key = video_id (crawl lại cùng video vào cùng partition).
ensure_topic "${KAFKA_TOPIC_VIDEOS:-tiktok.videos}"

# Giai đoạn 2 — task discover → Bronze writer, key = username. Bỏ comment khi làm discover.
# ensure_topic "${KAFKA_TOPIC_CREATORS:-tiktok.creators}"
