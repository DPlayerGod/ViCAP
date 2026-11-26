"""Bước ``profile`` (PLAN mục 4.3, 5): ``gold.video_predictions`` → ``gold.creator_profiles`` +
``gold.creator_profile_history``.

1. Spark đọc dự đoán. Có ``--batch-id`` thì chỉ tính lại creator có video trong các lô đó, nhưng hồ sơ vẫn
   tính trên **mọi** video của creator (``vicap.gold.profile.build_profiles``).
2. Kiểm định ``gold_profiles_suite`` bằng PyArrow: lỗi cấp lô → dừng; dòng lỗi → ``ops.quarantine`` stage
   ``gold_creator_profiles`` (``video_id`` null, gộp theo ``(username, lỗi)``; creator lần này qua được thì bỏ).
3. Dòng hợp lệ → ``MERGE INTO gold.creator_profiles`` theo ``username`` và ``gold.creator_profile_history``
   theo ``(snapshot_date, username)``. ``snapshot_date`` = ngày xử lý. Chỉ ghi dòng đổi; chạy lại thì không có gì.

Chạy trong container Airflow (``make up-spark``)::

    docker compose exec airflow-scheduler python -m vicap.pipelines.profile
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pyarrow as pa
from loguru import logger
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.config import IcebergConfig
from vicap.iceberg.tables import (
    GOLD_CREATOR_PROFILE_HISTORY,
    GOLD_CREATOR_PROFILES,
    GOLD_VIDEO_PREDICTIONS,
    OPS_QUARANTINE,
)
from vicap.iceberg.writer import ensure_table
from vicap.quality import gold_profiles_suite, validate
from vicap.quality.quarantine import quarantine_by_batch
from vicap.utils.dates import now_utc, parse_date
from vicap.utils.log import setup_logging

STAGE = "gold_creator_profiles"


@dataclass
class ProfileStats:
    videos: int = 0
    creators: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    history: int = 0  # dòng history ghi mới / cập nhật
    quarantined: int = 0

    def summary(self) -> str:
        return (f"profile: {self.videos} video → {self.creators} creator; thêm {self.inserted}, cập nhật "
                f"{self.updated}, giữ nguyên {self.unchanged}, history {self.history}, cách ly {self.quarantined}")


def run_profile(
    batch_ids: Sequence[str] | None = None,
    *,
    processing_date: str | date | datetime | None = None,
    spark: Any = None,
    catalog: Any = None,
    now: Callable[[], datetime] = now_utc,
    predictions_table: str = GOLD_VIDEO_PREDICTIONS.identifier,
    profiles_table: str = GOLD_CREATOR_PROFILES.identifier,
    history_table: str = GOLD_CREATOR_PROFILE_HISTORY.identifier,
    quarantine_table: str = OPS_QUARANTINE.identifier,
) -> ProfileStats:
    from pyspark.sql import functions as F

    from vicap.gold.profile import as_history, build_profiles, merge_changed
    from vicap.spark import get_spark

    updated_at = now()
    pdate = parse_date(processing_date or updated_at)
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    ensure_table(catalog, predictions_table, GOLD_VIDEO_PREDICTIONS.schema)
    prof_tbl = ensure_table(catalog, profiles_table, GOLD_CREATOR_PROFILES.schema)
    ensure_table(catalog, history_table, GOLD_CREATOR_PROFILE_HISTORY.schema,
                 GOLD_CREATOR_PROFILE_HISTORY.partition_spec)

    own_spark = spark is None
    spark = spark or get_spark("profile")
    prefix = IcebergConfig().catalog_name
    try:
        preds = spark.table(f"{prefix}.{predictions_table}")
        if batch_ids:
            users = preds.filter(F.col("batch_id").isin(list(batch_ids))).select("username").distinct()
            preds = preds.join(users, "username", "left_semi")
        preds = preds.cache()
        stats = ProfileStats(videos=preds.count())
        if not stats.videos:
            logger.info(stats.summary())
            return stats

        profiles = build_profiles(preds, updated_at).cache()
        schema = schema_to_pyarrow(prof_tbl.schema())
        checked = pa.Table.from_pylist([r.asDict() for r in profiles.collect()], schema=schema)
        stats.creators = checked.num_rows

        res = validate(checked, gold_profiles_suite())
        batches = sorted(set(checked.column("batch_id").to_pylist()))
        res.batch.quarantine_and_raise(checked, domain=STAGE, processing_date=pdate.isoformat(),
                                       batch_id=",".join(batches))
        bad_users = res.bad.column("username").to_pylist()
        good = profiles.filter(~F.col("username").isin(bad_users)) if bad_users else profiles

        counts = merge_changed(spark, good, f"{prefix}.{profiles_table}", ("username",))
        hist = merge_changed(spark, as_history(good, pdate), f"{prefix}.{history_table}",
                             ("snapshot_date", "username"))
        stats.inserted, stats.updated, stats.unchanged = counts.inserted, counts.updated, counts.unchanged
        stats.history = hist.inserted + hist.updated

        problems = [(r["batch_id"], None, r["username"], "; ".join(r["_errors"]))
                    for r in res.bad.select(["batch_id", "username", "_errors"]).to_pylist()]
        resolved: dict[str, set[str]] = defaultdict(set)
        for r in res.ok.select(["batch_id", "username"]).to_pylist():
            resolved[r["batch_id"]].add(r["username"])
        quarantine = ensure_table(catalog, quarantine_table, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
        quarantine_by_batch(quarantine, problems, stage=STAGE, processing_date=pdate, resolved=resolved)
        for b, _, u, err in problems:
            logger.warning("cách ly [{} {}] {}: {}", STAGE, b, u, err)
        stats.quarantined = len(problems)
        profiles.unpersist()
        preds.unpersist()
        logger.info(stats.summary())
        return stats
    finally:
        if own_spark:
            spark.stop()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-id", action="append", default=None,
                    help="chỉ tính lại creator có video trong lô này (lặp được); mặc định mọi creator")
    ap.add_argument("--processing-date", default=None, help="YYYY-MM-DD = snapshot_date, mặc định hôm nay (UTC)")
    args = ap.parse_args(argv)

    setup_logging()
    print(run_profile(args.batch_id, processing_date=args.processing_date).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
