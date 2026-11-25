"""Bước ``silver_clean`` (PLAN mục 5): ``bronze.videos_raw`` → ``silver.videos`` + ``ops.quarantine``.

1. Spark đọc Bronze (các ``batch_id`` được chọn, mặc định cả bảng), giữ bản crawl mới nhất mỗi video, dựng
   ``caption`` / ``hashtags`` như dataset v5 (``vicap.silver``).
2. Kiểm định ``silver_videos_suite`` bằng PyArrow: lỗi cấp lô → dừng; dòng lỗi → ``ops.quarantine`` stage
   ``silver_videos`` (gộp theo ``(stage, batch_id)``, video lần này qua được thì bỏ khỏi quarantine).
3. Dòng hợp lệ → ``MERGE INTO silver.videos`` theo ``video_id``.

Spark cần Java + jar Iceberg nên chạy trong container Airflow (cần ``make up-spark`` hoặc profile ``core pipeline``)::

    docker compose exec airflow-scheduler python -m vicap.pipelines.silver
    docker compose exec airflow-scheduler python -m vicap.pipelines.silver --batch-id 20261125-143314
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

from vicap.config import IcebergConfig
from vicap.iceberg.tables import BRONZE_VIDEOS_RAW, OPS_QUARANTINE, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table
from vicap.quality import silver_videos_suite, validate
from vicap.quality.quarantine import BAD_SCHEMA, merge_quarantine_rows
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging

STAGE = "silver_videos"


@dataclass
class SilverStats:
    bronze_rows: int = 0
    videos: int = 0  # video khác nhau sau khi giữ bản mới nhất
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    quarantined: int = 0
    batches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"silver_clean: {self.bronze_rows} dòng Bronze (lô {', '.join(self.batches) or '-'}) "
            f"→ {self.videos} video; "
            f"thêm {self.inserted}, cập nhật {self.updated}, giữ nguyên {self.unchanged}, cách ly {self.quarantined}"
        )


def _quarantine(catalog: Any, checked: pa.Table, bad: pa.Table, *, identifier: str, processing_date: date) -> None:
    """Gộp dòng lỗi theo ``batch_id``; video hợp lệ của lô thì đánh dấu đã xử lý (bỏ khỏi quarantine)."""
    bad_ids = set(bad.column("video_id").to_pylist())
    resolved: dict[str, set[str]] = defaultdict(set)
    for b, v in zip(checked.column("batch_id").to_pylist(), checked.column("video_id").to_pylist(), strict=True):
        if v not in bad_ids:
            resolved[b].add(v)
    rows: dict[str, list[dict]] = defaultdict(list)
    for r in bad.select(["video_id", "username", "batch_id", "_errors"]).to_pylist():
        rows[r.pop("batch_id")].append(r)
    table = ensure_table(catalog, identifier, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
    for b in sorted(set(rows) | set(resolved)):
        merge_quarantine_rows(table, pa.Table.from_pylist(rows.get(b, []), schema=BAD_SCHEMA), stage=STAGE,
                              batch_id=b, processing_date=processing_date, resolved=resolved.get(b, ()))
    for r in bad.select(["batch_id", "video_id", "_errors"]).to_pylist():
        logger.warning("cách ly [{} {}] {}: {}", STAGE, r["batch_id"], r["video_id"], "; ".join(r["_errors"]))


def run_silver_clean(
    batch_ids: Sequence[str] | None = None,
    *,
    processing_date: str | date | datetime | None = None,
    spark: Any = None,
    catalog: Any = None,
    now: Callable[[], datetime] = now_utc,
    bronze_table: str = BRONZE_VIDEOS_RAW.identifier,
    silver_table: str = SILVER_VIDEOS.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> SilverStats:
    from pyspark.sql import functions as F

    from vicap.silver.videos import build_silver, merge_into_silver
    from vicap.spark import get_spark

    updated_at = now()
    pdate = parse_date(processing_date or updated_at)
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    ensure_table(catalog, silver_table, SILVER_VIDEOS.schema, SILVER_VIDEOS.partition_spec)

    own_spark = spark is None
    spark = spark or get_spark("silver_clean")
    prefix = IcebergConfig().catalog_name
    try:
        bronze = spark.table(f"{prefix}.{bronze_table}")
        if batch_ids:
            bronze = bronze.filter(F.col("batch_id").isin(list(batch_ids)))
        stats = SilverStats(bronze_rows=bronze.count())
        stats.batches = sorted(r.batch_id for r in bronze.select("batch_id").distinct().collect())
        if stats.bronze_rows == 0:
            logger.info("silver_clean: không có dòng Bronze nào (lô {})", list(batch_ids or []) or "tất cả")
            return stats

        src = build_silver(bronze, updated_at).cache()
        suite = silver_videos_suite()
        cols = sorted({*suite.required, "video_id", "username", "batch_id"})
        checked = pa.Table.from_pylist([r.asDict() for r in src.select(*cols).collect()])
        stats.videos = checked.num_rows

        res = validate(checked, suite)
        res.batch.quarantine_and_raise(checked, domain=STAGE, processing_date=pdate.isoformat(),
                                       batch_id=",".join(stats.batches))
        bad_ids = res.bad.column("video_id").to_pylist()
        good = src.filter(~F.col("video_id").isin(bad_ids)) if bad_ids else src
        counts = merge_into_silver(spark, good, f"{prefix}.{silver_table}")
        stats.inserted, stats.updated, stats.unchanged = counts.inserted, counts.updated, counts.unchanged
        stats.quarantined = len(bad_ids)
        _quarantine(catalog, checked, res.bad, identifier=quarantine_table, processing_date=pdate)
        src.unpersist()
        logger.info(stats.summary())
        return stats
    finally:
        if own_spark:
            spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", action="append", default=None,
                    help="lô Bronze cần xử lý (lặp được); mặc định cả bảng")
    ap.add_argument("--processing-date", default=None, help="YYYY-MM-DD cho ops.quarantine, mặc định hôm nay (UTC)")
    args = ap.parse_args(argv)

    setup_logging()
    print(run_silver_clean(args.batch_id, processing_date=args.processing_date).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
