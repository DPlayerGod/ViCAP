"""silver_clean trên Spark + Iceberg REST thật, trong namespace tạm. Cần Java + jar Iceberg nên chạy trong
container Airflow: ``make test-spark``. Trên máy host (không có pyspark) test tự bỏ qua."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

pytest.importorskip("pyspark")

from pyiceberg.io.pyarrow import schema_to_pyarrow  # noqa: E402

from test_bronze import PDATE, T0, info_reader, ok_msg, rec  # noqa: E402
from vicap.bronze.videos import build_rows, decode, dedupe_latest  # noqa: E402
from vicap.iceberg.reader import read_table  # noqa: E402
from vicap.iceberg.tables import BRONZE_VIDEOS_RAW, OPS_QUARANTINE, SILVER_VIDEOS  # noqa: E402
from vicap.iceberg.writer import write_table  # noqa: E402
from vicap.pipelines.silver import STAGE, run_silver_clean  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.spark]


@pytest.fixture(scope="module")
def spark():
    from vicap.spark import get_spark

    s = get_spark("test_silver")
    yield s
    s.stop()


def _bronze_rows(table, *msgs):
    items = dedupe_latest(decode([rec(m, i) for i, m in enumerate(msgs)]).items)
    rows, problems = build_rows(items, info_reader(), processing_date=PDATE, ingested_at=T0,
                                schema=schema_to_pyarrow(table.schema()))
    assert not problems
    return rows


def test_silver_clean_latest_merge_and_quarantine(spark, iceberg_scratch) -> None:
    bronze = iceberg_scratch("videos_raw", BRONZE_VIDEOS_RAW)
    silver = iceberg_scratch("videos", SILVER_VIDEOS)
    quarantine = iceberg_scratch("quarantine", OPS_QUARANTINE)
    ns = bronze.name()[-2]
    names = dict(bronze_table=f"{ns}.videos_raw", silver_table=f"{ns}.videos", quarantine_table=f"{ns}.quarantine")

    def run(*batches: str):
        return run_silver_clean(list(batches) or None, spark=spark, processing_date=PDATE,
                                now=lambda: datetime(2026, 11, 25, 15, tzinfo=UTC), **names)

    desc = "Món ngon \n#Ăn_vặt hôm nay #food #food"
    a1 = ok_msg("1", "b1", plays=10)
    bad = replace(ok_msg("2", "b1"), duration=-5)  # Positive(duration) → cách ly
    write_table(bronze, _bronze_rows(bronze, a1, bad))
    write_table(bronze, _bronze_rows(bronze, ok_msg("1", "b2", minutes=5, plays=50)))

    s = run()
    assert (s.bronze_rows, s.videos, s.inserted, s.quarantined) == (3, 2, 1, 1)
    rows = {r["video_id"]: r for r in read_table(silver.refresh()).to_pylist()}
    assert list(rows) == ["1"]
    assert rows["1"]["batch_id"] == "b2" and rows["1"]["play_count"] == 50  # bản crawl mới nhất
    assert rows["1"]["caption"] == "hello" and rows["1"]["hashtags"] == ["food"]
    q = read_table(quarantine.refresh()).to_pylist()
    assert [(r["stage"], r["batch_id"], r["video_id"]) for r in q] == [(STAGE, "b1", "2")]

    # chạy lại / chạy lô cũ: không đè bản mới hơn, không snapshot mới
    snaps = len(silver.refresh().snapshots())
    assert (run().unchanged, run("b1").unchanged) == (1, 1)
    assert len(silver.refresh().snapshots()) == snaps
    assert read_table(silver.refresh()).to_pylist()[0]["batch_id"] == "b2"

    # sửa video 2 ở lô b1 → vào Silver, bỏ khỏi quarantine
    fixed = replace(ok_msg("2", "b1"), description=desc)
    write_table(bronze, _bronze_rows(bronze, fixed), mode="overwrite", overwrite_filter="video_id == '2'")
    s = run("b1")
    assert (s.inserted, s.quarantined) == (1, 0)
    row2 = next(r for r in read_table(silver.refresh()).to_pylist() if r["video_id"] == "2")
    assert row2["caption"] == "Món ngon \n hôm nay" and row2["hashtags"] == ["Ăn_vặt", "food", "food"]
    assert read_table(quarantine.refresh()).num_rows == 0
