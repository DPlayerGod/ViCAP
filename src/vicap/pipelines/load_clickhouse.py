"""Bước ``load_clickhouse`` (PLAN mục 3, 5): 3 bảng gold Iceberg → ClickHouse ``vicap.*`` cho dashboard.

| Iceberg                         | ClickHouse                       | version (ReplacingMergeTree) |
|---------------------------------|----------------------------------|------------------------------|
| ``gold.video_predictions``      | ``vicap.video_predictions``      | ``predicted_at``             |
| ``gold.creator_profiles``       | ``vicap.creator_profiles``       | ``updated_at``               |
| ``gold.creator_profile_history``| ``vicap.creator_profile_history``| ``updated_at``               |

Nạp tăng dần: mốc = ``max(version)`` đang có trong ClickHouse, chỉ nạp dòng Iceberg có version lớn hơn mốc
(mỗi lần ``predict`` / ``profile`` ghi đều đặt version = thời điểm chạy). Chạy lại ngay thì không nạp gì. Dòng cùng
khoá được ReplacingMergeTree gộp, giữ version lớn nhất; truy vấn dùng ``FINAL``. ``--full`` nạp lại toàn bộ rồi
``OPTIMIZE … FINAL`` (vd sau khi tạo lại volume ClickHouse).

Chạy trong container Airflow hoặc trên máy host (cần ClickHouse, ``make up-serving``)::

    docker compose exec airflow-scheduler python -m vicap.pipelines.load_clickhouse
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from vicap.config import ClickHouseConfig
from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import (
    GOLD_CREATOR_PROFILE_HISTORY,
    GOLD_CREATOR_PROFILES,
    GOLD_VIDEO_PREDICTIONS,
    TableDef,
)
from vicap.utils.log import setup_logging

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Target:
    source: TableDef
    table: str  # tên bảng trong database ClickHouse
    version: str


TARGETS = (
    Target(GOLD_VIDEO_PREDICTIONS, "video_predictions", "predicted_at"),
    Target(GOLD_CREATOR_PROFILES, "creator_profiles", "updated_at"),
    Target(GOLD_CREATOR_PROFILE_HISTORY, "creator_profile_history", "updated_at"),
)


@dataclass
class LoadStats:
    loaded: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        return "load_clickhouse: " + ", ".join(f"{t} +{n}" for t, n in self.loaded.items())


def watermark(client: Any, table: str, version: str) -> datetime:
    """``max(version)`` trong ClickHouse; bảng trống → 1970-01-01 (ClickHouse trả epoch cho cột không Nullable)."""
    value = client.query(f"SELECT max({version}) FROM {table}").result_rows[0][0]
    if value is None:
        return EPOCH
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def rows_after(table: pa.Table, version: str, mark: datetime) -> pa.Table:
    if mark <= EPOCH:
        return table
    return table.filter(pc.greater(table.column(version), pa.scalar(mark, pa.timestamp("us", tz="UTC"))))


def run_load_clickhouse(
    *,
    full: bool = False,
    client: Any = None,
    catalog: Any = None,
    database: str | None = None,
    targets: Sequence[Target] = TARGETS,
    sources: dict[str, str] | None = None,
) -> LoadStats:
    """``sources``: đổi identifier Iceberg của từng bảng ClickHouse (test dùng namespace tạm)."""
    if catalog is None:
        from vicap.iceberg.catalog import load_lakehouse_catalog

        catalog = load_lakehouse_catalog()
    if client is None:
        from vicap.clickhouse.client import get_clickhouse_client

        client = get_clickhouse_client()
    database = database or ClickHouseConfig().database
    sources = sources or {}

    stats = LoadStats()
    for t in targets:
        name = f"{database}.{t.table}"
        identifier = sources.get(t.table, t.source.identifier)
        if not catalog.table_exists(identifier):
            stats.loaded[t.table] = 0
            continue
        src = read_table(catalog.load_table(identifier))
        mark = EPOCH if full else watermark(client, name, t.version)
        new = rows_after(src, t.version, mark)
        if new.num_rows:
            client.insert_arrow(name, new.select([f.name for f in t.source.schema.fields]))
            if full:
                client.command(f"OPTIMIZE TABLE {name} FINAL")
        stats.loaded[t.table] = new.num_rows
        logger.info("{} → {}: {} dòng (mốc {})", identifier, name, new.num_rows, mark.isoformat())
    logger.info(stats.summary())
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="nạp lại toàn bộ rồi OPTIMIZE FINAL")
    args = ap.parse_args(argv)

    setup_logging()
    print(run_load_clickhouse(full=args.full).summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
