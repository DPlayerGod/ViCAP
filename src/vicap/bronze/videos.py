"""Bronze ``bronze.videos_raw`` từ topic ``tiktok.videos`` (PLAN mục 5; TODO Pipeline).

Hàm thuần, không gọi Kafka / MinIO / Iceberg (nhận ``read_info`` để đọc ``info.json``), để test trên máy:

1. ``decode``: value → ``VideoMessage``; message hỏng (không phải JSON, sai schema, key ≠ ``video_id``) tách riêng.
2. ``dedupe_latest``: khử trùng theo ``(batch_id, video_id)``, giữ ``crawl_ts`` mới nhất (retry của task crawl
   gửi lại cả lô). Làm trên cả message ``ok`` lẫn ``error``: lần thử lại thành công thì thay lần lỗi.
3. ``build_rows``: message ``ok`` + ``info.json`` → dòng Bronze; ``info.json`` đọc không được thì tách riêng.
4. ``merge_rows``: gộp với các dòng đã có của cùng ``batch_id``, rồi ghi đè theo ``batch_id`` (một commit).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

import pyarrow as pa

from vicap.ingestion.message import MessageError, VideoMessage


@dataclass(frozen=True)
class KafkaRecord:
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None


@dataclass(frozen=True)
class Item:
    record: KafkaRecord
    message: VideoMessage

    @property
    def order(self) -> tuple[datetime, int, int]:
        return self.message.crawl_ts, self.record.partition, self.record.offset


@dataclass(frozen=True)
class Problem:
    """Một dòng sẽ vào ``ops.quarantine``."""

    stage: str
    batch_id: str
    video_id: str | None
    username: str | None
    error: str


STAGE_CRAWL = "crawl"  # message status="error": crawler không tải được bài
STAGE_BRONZE = "bronze_videos"  # message hỏng, info.json không đọc được
INVALID_BATCH = "kafka-invalid"  # message hỏng không đọc được batch_id


@dataclass
class Decoded:
    items: list[Item] = field(default_factory=list)
    invalid: list[Problem] = field(default_factory=list)


def _where(record: KafkaRecord) -> str:
    return f"kafka p{record.partition}/o{record.offset}"


def decode(records: Iterable[KafkaRecord]) -> Decoded:
    out = Decoded()
    for rec in records:
        key = rec.key.decode("utf-8", "replace") if rec.key is not None else None
        try:
            if rec.value is None:
                raise MessageError("value rỗng")
            msg = VideoMessage.from_bytes(rec.value)
            if key != msg.video_id:
                raise MessageError(f"key {key!r} khác video_id {msg.video_id!r}")
        except MessageError as e:
            vid = key if key and key.isdigit() else None
            out.invalid.append(Problem(STAGE_BRONZE, INVALID_BATCH, vid, None, f"message_invalid: {e} ({_where(rec)})"))
            continue
        out.items.append(Item(rec, msg))
    return out


def dedupe_latest(items: Iterable[Item]) -> list[Item]:
    """Mỗi ``(batch_id, video_id)`` giữ message có ``crawl_ts`` mới nhất (bằng nhau thì offset sau)."""
    best: dict[tuple[str, str], Item] = {}
    for it in items:
        k = (it.message.batch_id, it.message.video_id)
        if k not in best or it.order > best[k].order:
            best[k] = it
    return sorted(best.values(), key=lambda it: (it.message.batch_id, it.message.video_id))


def crawl_problems(items: Iterable[Item]) -> list[Problem]:
    return [
        Problem(STAGE_CRAWL, it.message.batch_id, it.message.video_id, it.message.username,
                f"crawl_failed: {it.message.error}")
        for it in items if not it.message.ok
    ]


def build_rows(
    items: Sequence[Item],
    read_info: Callable[[str], bytes],
    *,
    processing_date: date,
    ingested_at: datetime,
    schema: pa.Schema,
) -> tuple[pa.Table, list[Problem]]:
    """Dòng Bronze cho các message ``ok``; ``read_info(info_key)`` trả bytes của ``info.json``."""
    rows, problems = [], []
    for it in items:
        m = it.message
        if not m.ok:
            continue
        try:
            raw_json = read_info(m.info_key).decode("utf-8")
        except Exception as e:  # noqa: BLE001 — thiếu file / hỏng UTF-8 / mất quyền: cách ly dòng này
            problems.append(Problem(STAGE_BRONZE, m.batch_id, m.video_id, m.username,
                                    f"info_json_unreadable: {type(e).__name__}: {e}"[:500]))
            continue
        row = {name: getattr(m, name) for name in schema.names if hasattr(m, name)}
        row.update(
            raw_json=raw_json,
            kafka_partition=it.record.partition,
            kafka_offset=it.record.offset,
            processing_date=processing_date,
            ingested_at=ingested_at,
        )
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema), problems


def merge_rows(existing: pa.Table | None, new: pa.Table) -> pa.Table:
    """Gộp dòng cũ và mới của cùng các ``batch_id``; mỗi ``(batch_id, video_id)`` giữ ``crawl_ts`` mới nhất.

    Bằng ``crawl_ts`` thì dòng mới thắng (vd chạy lại Bronze sau khi sửa lỗi đọc ``info.json``).
    """
    if existing is None or existing.num_rows == 0:
        tables = [new]
    else:
        tables = [existing.select(new.schema.names).cast(new.schema), new]
    best: dict[tuple[str, str], tuple[tuple[datetime, int], int, int]] = {}
    for src, t in enumerate(tables):
        cols = t.select(["batch_id", "video_id", "crawl_ts"]).to_pydict()
        for i, (b, v, ts) in enumerate(zip(cols["batch_id"], cols["video_id"], cols["crawl_ts"], strict=True)):
            rank = (ts, src)
            if (b, v) not in best or rank >= best[(b, v)][0]:
                best[(b, v)] = (rank, src, i)
    picked = sorted(best.items())
    parts = [
        tables[src].take(pa.array([i for _, (_, s, i) in picked if s == src], pa.int64()))
        for src in range(len(tables))
    ]
    merged = pa.concat_tables(parts)
    return merged.sort_by([("batch_id", "ascending"), ("video_id", "ascending")])
