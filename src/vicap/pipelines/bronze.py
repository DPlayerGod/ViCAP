"""Bước ``bronze`` (PLAN mục 5): ``tiktok.videos`` → ``bronze.videos_raw`` + ``ops.quarantine``.

1. Đọc Kafka từ offset đã commit tới ``end_offsets`` lúc bắt đầu (``KafkaSource``).
2. Message → ``VideoMessage``, khử trùng theo ``(batch_id, video_id)`` giữ ``crawl_ts`` mới nhất.
3. Message ``ok``: đọc ``info.json`` từ ``raw/`` vào ``raw_json``; gộp với dòng đã có của cùng ``batch_id`` rồi
   ghi đè theo ``batch_id`` (một commit Iceberg).
4. Cách ly: message hỏng / ``info.json`` không đọc được → stage ``bronze_videos``; message ``status="error"``
   → stage ``crawl``. Video lần này đã vào Bronze được thì bỏ khỏi quarantine của lô đó.
5. Ghi xong mới commit offset Kafka.

Chạy tay (máy host, cần ``make up PROFILE="core stream"``)::

    uv run python -m vicap.pipelines.bronze
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import pyarrow as pa
from loguru import logger
from pyiceberg.expressions import In
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.bronze.source import KafkaSource
from vicap.bronze.videos import (
    STAGE_BRONZE,
    STAGE_CRAWL,
    KafkaRecord,
    Problem,
    build_rows,
    crawl_problems,
    decode,
    dedupe_latest,
    merge_rows,
)
from vicap.config import MinioConfig
from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import BRONZE_VIDEOS_RAW, OPS_QUARANTINE
from vicap.iceberg.writer import ensure_table, write_table
from vicap.quality.quarantine import BAD_SCHEMA, merge_quarantine_rows
from vicap.storage.objects import s3_filesystem
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging


@dataclass
class BronzeStats:
    records: int = 0  # record Kafka đã đọc
    invalid: int = 0  # message hỏng
    duplicates: int = 0  # bị khử trùng trong lần đọc này
    rows: int = 0  # dòng Bronze mới ghi
    crawl_errors: int = 0
    info_errors: int = 0
    batches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"bronze: đọc {self.records} message, ghi {self.rows} dòng (lô {', '.join(self.batches) or '-'}), "
            f"trùng {self.duplicates}, hỏng {self.invalid}, crawl lỗi {self.crawl_errors}, "
            f"info.json lỗi {self.info_errors}"
        )


def _bad_table(problems: Sequence[Problem]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"video_id": p.video_id, "username": p.username, "_errors": [p.error]} for p in problems], schema=BAD_SCHEMA
    )


def write_bronze(
    records: Sequence[KafkaRecord],
    *,
    catalog: Any,
    read_info: Callable[[str], bytes],
    processing_date: date,
    ingested_at: datetime,
) -> BronzeStats:
    """Phần ghi của ``run_bronze`` (không đụng Kafka)."""
    stats = BronzeStats(records=len(records))
    decoded = decode(records)
    items = dedupe_latest(decoded.items)
    stats.invalid = len(decoded.invalid)
    stats.duplicates = len(decoded.items) - len(items)

    bronze = ensure_table(catalog, BRONZE_VIDEOS_RAW.identifier, BRONZE_VIDEOS_RAW.schema,
                          BRONZE_VIDEOS_RAW.partition_spec)
    new, info_problems = build_rows(
        items, read_info, processing_date=processing_date, ingested_at=ingested_at,
        schema=schema_to_pyarrow(bronze.schema()),
    )
    resolved: dict[str, set[str]] = defaultdict(set)  # batch_id → video đã vào Bronze ở lần này
    if new.num_rows:
        batches = sorted(set(new.column("batch_id").to_pylist()))
        existing = read_table(bronze, In("batch_id", batches))
        write_table(bronze, merge_rows(existing, new), mode="overwrite", overwrite_filter=In("batch_id", batches))
        for b, v in zip(new.column("batch_id").to_pylist(), new.column("video_id").to_pylist(), strict=True):
            resolved[b].add(v)
        stats.rows, stats.batches = new.num_rows, batches

    # Message lỗi của một video đã có dòng Bronze cùng lô (lần thử trước thành công) thì không cách ly.
    errors = crawl_problems(items)
    if errors:
        in_bronze = read_table(bronze, In("batch_id", sorted({p.batch_id for p in errors})))
        have = set(zip(in_bronze.column("batch_id").to_pylist(), in_bronze.column("video_id").to_pylist(),
                       strict=True))
        errors = [p for p in errors if (p.batch_id, p.video_id) not in have]
    stats.crawl_errors, stats.info_errors = len(errors), len(info_problems)

    problems = [*decoded.invalid, *errors, *info_problems]
    groups: dict[tuple[str, str], list[Problem]] = defaultdict(list)
    for p in problems:
        groups[(p.stage, p.batch_id)].append(p)
    for b in resolved:
        for stage in (STAGE_CRAWL, STAGE_BRONZE):
            groups.setdefault((stage, b), [])
    if groups:
        quarantine = ensure_table(catalog, OPS_QUARANTINE.identifier, OPS_QUARANTINE.schema,
                                  OPS_QUARANTINE.partition_spec)
        for (stage, b), ps in sorted(groups.items()):
            merge_quarantine_rows(quarantine, _bad_table(ps), stage=stage, batch_id=b,
                                  processing_date=processing_date, resolved=resolved.get(b, ()))
    for p in problems:
        logger.warning("cách ly [{} {}] {}: {}", p.stage, p.batch_id, p.video_id, p.error[:200])
    return stats


def run_bronze(
    processing_date: str | date | datetime | None = None,
    *,
    catalog: Any = None,
    source: KafkaSource | None = None,
    raw_fs: Any = None,
    now: Callable[[], datetime] = now_utc,
) -> BronzeStats:
    ingested_at = now()
    pdate = parse_date(processing_date or ingested_at)
    own_source = source is None
    source = source or KafkaSource()
    try:
        records = source.read_pending()
        if not records:
            logger.info("bronze: không có message mới")
            return BronzeStats()
        if catalog is None:
            from vicap.iceberg.catalog import load_lakehouse_catalog

            catalog = load_lakehouse_catalog()
        config = MinioConfig()
        fs = raw_fs or s3_filesystem(config)
        stats = write_bronze(
            records, catalog=catalog, read_info=lambda key: fs.cat_file(f"{config.raw_bucket}/{key}"),
            processing_date=pdate, ingested_at=ingested_at,
        )
        source.commit()  # chỉ sau khi Iceberg đã ghi xong
        logger.info(stats.summary())
        return stats
    finally:
        if own_source:
            source.close()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processing-date", default=None, help="YYYY-MM-DD, mặc định hôm nay (UTC)")
    args = ap.parse_args(argv)

    setup_logging()
    print(run_bronze(args.processing_date).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
