from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pyarrow as pa
import pytest

from vicap.bronze.source import KafkaSource
from vicap.bronze.videos import (
    INVALID_BATCH,
    STAGE_BRONZE,
    STAGE_CRAWL,
    KafkaRecord,
    build_rows,
    decode,
    dedupe_latest,
    merge_rows,
)
from vicap.iceberg.tables import BRONZE_VIDEOS_RAW_SCHEMA
from vicap.ingestion.message import VideoMessage
from vicap.pipelines.bronze import write_bronze

T0 = datetime(2026, 11, 25, 14, 0, tzinfo=UTC)
PDATE = date(2026, 11, 25)


def ok_msg(vid: str, batch: str = "b1", *, minutes: int = 0, plays: int = 100, slideshow: bool = False) -> VideoMessage:
    prefix = f"videos/{vid}/"
    return VideoMessage(
        video_id=vid, username="creator.a", batch_id=batch, crawl_ts=T0 + timedelta(minutes=minutes),
        media_type="slideshow" if slideshow else "video", posted_at=T0 - timedelta(days=1),
        description="hello #food", hashtags=["food"], duration=30, width=576, height=720,
        play_count=plays, like_count=5, comment_count=1, share_count=0, save_count=2, follower_count=1000,
        info_key=prefix + "info.json", media_key=prefix if slideshow else prefix + "video.mp4",
        thumbnail_key=prefix + "cover.jpg", image_keys=[prefix + "img_001.jpg"] if slideshow else [],
    )


def err_msg(vid: str, batch: str = "b1", *, minutes: int = 0) -> VideoMessage:
    return VideoMessage.failed(vid, "creator.a", batch_id=batch, crawl_ts=T0 + timedelta(minutes=minutes),
                               error="status=10231")


def rec(msg: VideoMessage | None, offset: int, *, partition: int = 0, key: bytes | None = None,
        value: bytes | None = None) -> KafkaRecord:
    if msg is not None:
        key, value = key or msg.key(), msg.to_bytes()
    return KafkaRecord(partition, offset, key, value)


def info_reader(missing: set[str] = frozenset()):
    def read(key: str) -> bytes:
        vid = key.split("/")[1]
        if vid in missing:
            raise FileNotFoundError(key)
        return json.dumps({"id": vid, "desc": "xin chào"}, ensure_ascii=False).encode()

    return read


# ---------------------------------------------------------------- hàm thuần


def test_decode_separates_invalid() -> None:
    good = ok_msg("11")
    d = decode([
        rec(good, 0),
        rec(None, 1, key=b"12", value=b"not json"),
        rec(None, 2, key=b"13", value=None),
        rec(good, 3, key=b"99"),  # key khác video_id
        rec(None, 4, key=None, value=b'{"v": 2}'),
    ])
    assert [it.message.video_id for it in d.items] == ["11"]
    assert [(p.stage, p.batch_id, p.video_id) for p in d.invalid] == [
        (STAGE_BRONZE, INVALID_BATCH, "12"), (STAGE_BRONZE, INVALID_BATCH, "13"),
        (STAGE_BRONZE, INVALID_BATCH, "99"), (STAGE_BRONZE, INVALID_BATCH, None),
    ]
    assert all("message_invalid" in p.error and "kafka p0/o" in p.error for p in d.invalid)


def test_dedupe_keeps_latest_per_batch_and_video() -> None:
    d = decode([
        rec(err_msg("11"), 0),
        rec(ok_msg("11", minutes=5, plays=200), 1),  # retry thành công thay lần lỗi
        rec(ok_msg("11", minutes=1, plays=150), 2),  # cũ hơn dù offset sau
        rec(ok_msg("11", batch="b2", minutes=9), 3),  # lô khác: giữ riêng
    ])
    items = dedupe_latest(d.items)
    assert [(it.message.batch_id, it.message.play_count, it.message.ok) for it in items] == [
        ("b1", 200, True), ("b2", 100, True),
    ]


def _schema() -> pa.Schema:
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    return schema_to_pyarrow(BRONZE_VIDEOS_RAW_SCHEMA)


def test_build_rows_reads_info_and_isolates_missing() -> None:
    items = dedupe_latest(decode([rec(ok_msg("11"), 0), rec(ok_msg("12", slideshow=True), 1),
                                  rec(ok_msg("13"), 2), rec(err_msg("14"), 3)]).items)
    table, problems = build_rows(items, info_reader({"13"}), processing_date=PDATE, ingested_at=T0, schema=_schema())
    assert table.column("video_id").to_pylist() == ["11", "12"]
    assert json.loads(table.column("raw_json")[0].as_py()) == {"id": "11", "desc": "xin chào"}
    assert table.column("image_keys").to_pylist() == [[], ["videos/12/img_001.jpg"]]
    assert table.column("kafka_offset").to_pylist() == [0, 1]
    assert [(p.stage, p.video_id) for p in problems] == [(STAGE_BRONZE, "13")]
    assert problems[0].error.startswith("info_json_unreadable: FileNotFoundError")


def test_merge_rows_keeps_latest_crawl_ts() -> None:
    schema = _schema()

    def rows(*msgs: VideoMessage) -> pa.Table:
        items = dedupe_latest(decode([rec(m, i) for i, m in enumerate(msgs)]).items)
        return build_rows(items, info_reader(), processing_date=PDATE, ingested_at=T0, schema=schema)[0]

    existing = rows(ok_msg("11", minutes=5, plays=500), ok_msg("12", plays=1), ok_msg("13", plays=7))
    new = rows(ok_msg("11", minutes=1, plays=100), ok_msg("12", minutes=3, plays=300), ok_msg("14"))
    merged = merge_rows(existing, new)
    assert dict(zip(merged.column("video_id").to_pylist(), merged.column("play_count").to_pylist(), strict=True)) == {
        "11": 500, "12": 300, "13": 7, "14": 100,
    }
    assert merge_rows(None, new).num_rows == 3


# ---------------------------------------------------------------- ghi trên catalog sqlite


def _read(catalog, identifier: str) -> list[dict]:
    return catalog.load_table(identifier).scan().to_arrow().to_pylist()


def test_write_bronze_idempotent_and_quarantine(sql_catalog) -> None:
    run1 = [
        rec(ok_msg("11"), 0), rec(ok_msg("12", slideshow=True), 1), rec(err_msg("13"), 2),
        rec(ok_msg("14"), 3),  # info.json thiếu
        rec(None, 4, key=b"15", value=b"{broken"),
    ]
    stats = write_bronze(run1, catalog=sql_catalog, read_info=info_reader({"14"}), processing_date=PDATE,
                         ingested_at=T0)
    assert (stats.rows, stats.crawl_errors, stats.info_errors, stats.invalid) == (2, 1, 1, 1)
    bronze = _read(sql_catalog, "bronze.videos_raw")
    assert sorted(r["video_id"] for r in bronze) == ["11", "12"]
    q = {(r["stage"], r["batch_id"], r["video_id"]) for r in _read(sql_catalog, "ops.quarantine")}
    assert q == {(STAGE_CRAWL, "b1", "13"), (STAGE_BRONZE, "b1", "14"), (STAGE_BRONZE, INVALID_BATCH, "15")}

    # Airflow retry task crawl: gửi lại cả lô b1, lần này 13 và 14 đều được; 11 có lượt xem mới.
    run2 = [rec(ok_msg("11", minutes=10, plays=999), 5), rec(ok_msg("12", slideshow=True, minutes=10), 6),
            rec(ok_msg("13", minutes=10), 7), rec(ok_msg("14", minutes=10), 8)]
    stats = write_bronze(run2, catalog=sql_catalog, read_info=info_reader(), processing_date=PDATE, ingested_at=T0)
    assert stats.rows == 4
    bronze = {r["video_id"]: r for r in _read(sql_catalog, "bronze.videos_raw")}
    assert sorted(bronze) == ["11", "12", "13", "14"]  # không trùng
    assert bronze["11"]["play_count"] == 999
    q = {(r["stage"], r["batch_id"], r["video_id"]) for r in _read(sql_catalog, "ops.quarantine")}
    assert q == {(STAGE_BRONZE, INVALID_BATCH, "15")}  # 13, 14 đã vào Bronze → bỏ khỏi quarantine

    # Lô b2 riêng; message lỗi tới sau cho video đã có dòng Bronze cùng lô thì không cách ly.
    run3 = [rec(ok_msg("11", batch="b2", minutes=20), 9), rec(err_msg("11", batch="b1", minutes=30), 10)]
    stats = write_bronze(run3, catalog=sql_catalog, read_info=info_reader(), processing_date=PDATE, ingested_at=T0)
    assert (stats.rows, stats.crawl_errors) == (1, 0)
    bronze = _read(sql_catalog, "bronze.videos_raw")
    assert sorted((r["batch_id"], r["video_id"]) for r in bronze) == [
        ("b1", "11"), ("b1", "12"), ("b1", "13"), ("b1", "14"), ("b2", "11"),
    ]
    assert {r["stage"] for r in _read(sql_catalog, "ops.quarantine")} == {STAGE_BRONZE}


# ---------------------------------------------------------------- KafkaSource với consumer giả


class FakeRecord:
    def __init__(self, partition: int, offset: int):
        self.partition, self.offset, self.key, self.value = partition, offset, str(offset).encode(), b"{}"


class FakeConsumer:
    """2 partition; p0 có offset 0..4, p1 có 0..2. Đã commit p0 = 2. Có record tới sau end (p0 offset 5)."""

    def __init__(self) -> None:
        from kafka import TopicPartition

        self.tp = [TopicPartition("tiktok.videos", 0), TopicPartition("tiktok.videos", 1)]
        self.end = {self.tp[0]: 5, self.tp[1]: 3}
        self.log = {self.tp[0]: list(range(6)), self.tp[1]: list(range(3))}
        self.pos = {tp: 0 for tp in self.tp}
        self._committed = {self.tp[0]: 2}
        self.commits: list[dict] = []
        self.closed = False

    def topics(self):
        return {"tiktok.videos"}

    def partitions_for_topic(self, topic):
        return {0, 1}

    def assign(self, tps):
        assert list(tps) == self.tp

    def end_offsets(self, tps):
        return dict(self.end)

    def committed(self, tp):
        return self._committed.get(tp)

    def seek(self, tp, offset):
        self.pos[tp] = offset

    def position(self, tp):
        return self.pos[tp]

    def poll(self, timeout_ms):
        out = {}
        for tp in self.tp:
            offs = [o for o in self.log[tp] if o >= self.pos[tp]][:2]
            if offs:
                out[tp] = [FakeRecord(tp.partition, o) for o in offs]
                self.pos[tp] = offs[-1] + 1
        return out

    def commit(self, offsets):
        self.commits.append({tp.partition: om.offset for tp, om in offsets.items()})

    def close(self):
        self.closed = True


def test_kafka_source_reads_to_end_offsets_and_commits_them() -> None:
    from vicap.config import KafkaConfig

    consumer = FakeConsumer()
    with KafkaSource(KafkaConfig(), consumer=consumer) as src:
        records = src.read_pending()
        assert sorted((r.partition, r.offset) for r in records) == [(0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2)]
        src.commit()
    assert consumer.commits == [{0: 5, 1: 3}]
    assert consumer.closed


def test_kafka_source_missing_topic() -> None:
    from vicap.config import KafkaConfig

    consumer = FakeConsumer()
    consumer.partitions_for_topic = lambda topic: None
    with pytest.raises(RuntimeError, match="không tồn tại"):
        KafkaSource(KafkaConfig(), consumer=consumer).read_pending()
