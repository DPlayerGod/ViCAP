"""KafkaSource trên Kafka thật (topic tạm + group tạm): đọc tới end_offsets, commit, lần sau chỉ đọc message mới."""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.integration


def test_kafka_source_commit_and_resume(kafka_topic) -> None:
    from kafka import KafkaProducer

    from vicap.bronze.source import KafkaSource
    from vicap.config import KafkaConfig

    config = KafkaConfig(topic=kafka_topic, group_id=f"it-bronze-{uuid.uuid4().hex[:8]}")
    producer = KafkaProducer(bootstrap_servers=config.bootstrap_servers.split(","), acks="all")

    def send(*keys: str) -> None:
        for k in keys:
            producer.send(kafka_topic, key=k.encode(), value=f'{{"k":"{k}"}}'.encode())
        producer.flush()

    def read() -> list[str]:
        with KafkaSource(config) as src:
            keys = sorted(r.key.decode() for r in src.read_pending())
            src.commit()
        return keys

    try:
        send("1", "2", "3", "4", "5")
        assert read() == ["1", "2", "3", "4", "5"]
        assert read() == []  # đã commit
        send("6", "7")
        assert read() == ["6", "7"]

        # Không commit (vd ghi Iceberg lỗi) → lần sau đọc lại đúng các message đó.
        send("8")
        with KafkaSource(config) as src:
            assert [r.key for r in src.read_pending()] == [b"8"]
        assert read() == ["8"]
    finally:
        producer.close()  # offset của group tạm bị Kafka xoá cùng topic tạm (fixture kafka_topic)
