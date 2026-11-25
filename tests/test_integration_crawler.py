"""Crawler trên MinIO + Kafka thật (`make up-crawl`), TikTok thay bằng client giả.

Ghi vào topic tạm (tạo rồi xoá trong test) để Bronze không đọc nhầm, và xoá file trên raw/ sau khi chạy.
"""

from __future__ import annotations

import socket
import uuid

import pytest

from test_ingestion import make_client  # tests/ nằm trong sys.path (pytest rootdir)

pytestmark = pytest.mark.integration


@pytest.fixture
def kafka_topic():
    from vicap.config import KafkaConfig

    host, port = KafkaConfig().bootstrap_servers.split(",")[0].rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
    except OSError:
        pytest.skip("Kafka chưa chạy — make up-crawl")
    from kafka.admin import KafkaAdminClient, NewTopic

    admin = KafkaAdminClient(bootstrap_servers=KafkaConfig().bootstrap_servers)
    topic = f"it.videos.{uuid.uuid4().hex[:8]}"
    admin.create_topics([NewTopic(topic, num_partitions=3, replication_factor=1)])
    try:
        yield topic
    finally:
        admin.delete_topics([topic])
        admin.close()


def test_crawl_to_minio_and_kafka(kafka_topic, monkeypatch) -> None:
    from kafka import KafkaConsumer

    from vicap.config import KafkaConfig
    from vicap.ingestion.crawler import KafkaSink, RawStore, crawl_creators
    from vicap.ingestion.message import VideoMessage

    # video_id ngẫu nhiên để không đụng dữ liệu thật trên raw/
    base = 9_000_000_000_000_000_000 + uuid.uuid4().int % 10**15
    vid, sid = str(base), str(base + 1)
    client = make_client()
    client.items = {vid: {**client.items["11"], "id": vid}, sid: {**client.items["22"], "id": sid}}
    client.posts = {"creator.a": [vid, sid]}
    client.files[f"https://cdn/mp4/{vid}"] = client.files.pop("https://cdn/mp4/11")

    monkeypatch.setenv("KAFKA_TOPIC_VIDEOS", kafka_topic)
    store = RawStore()
    try:
        with KafkaSink() as sink:
            stats = crawl_creators(["creator.a"], n=10, batch_id="it", client=client, store=store, send=sink,
                                   pause=lambda s: None)
        assert (stats.ok, stats.failed) == (2, 0)

        consumer = KafkaConsumer(kafka_topic, bootstrap_servers=KafkaConfig().bootstrap_servers,
                                 auto_offset_reset="earliest", enable_auto_commit=False, consumer_timeout_ms=5000)
        records = list(consumer)
        consumer.close()
        assert sorted(r.key.decode() for r in records) == [vid, sid]
        for r in records:
            m = VideoMessage.from_bytes(r.value)
            assert m.ok and m.video_id == r.key.decode()
            for key in (m.info_key, m.thumbnail_key, *m.image_keys):
                assert store.exists(key), key
        assert store.exists(f"videos/{vid}/video.mp4")
    finally:
        for v in (vid, sid):
            store.fs.rm(f"{store.bucket}/videos/{v}/", recursive=True)
