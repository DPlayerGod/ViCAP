"""vicap.quality.quarantine: dòng bị cách ly → bảng ops.quarantine (PLAN mục 9.3)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest

from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import OPS_QUARANTINE
from vicap.iceberg.writer import ensure_table
from vicap.quality import bronze_videos_suite, quarantine_rows, validate, write_quarantine_rows

TS = datetime(2026, 11, 25, tzinfo=UTC)


def _bad(ids: list[str]) -> pa.Table:
    return pa.table({
        "video_id": ids,
        "username": ["u"] * len(ids),
        "text_emb": pa.array([[[0.0]]] * len(ids), type=pa.list_(pa.list_(pa.float32()))),
        "_errors": [["non_zero_vector:text_emb[0]"]] * len(ids),
    })


def test_quarantine_rows_columns() -> None:
    rows = quarantine_rows(_bad(["123"]), stage="gold_features", batch_id="b1", processing_date="2026-11-25",
                           created_at=TS)
    assert rows.to_pylist() == [{
        "video_id": "123", "username": "u", "stage": "gold_features", "batch_id": "b1",
        "processing_date": date(2026, 11, 25), "errors": ["non_zero_vector:text_emb[0]"], "severity": "error",
        "raw_ref": "s3://raw/videos/123/", "resolved": False, "created_at": TS,
    }]


def test_profile_rows_have_no_video() -> None:
    bad = pa.table({"username": ["a"], "_errors": [["in_set:confidence"]]})
    row = quarantine_rows(bad, stage="gold_profiles", batch_id="b", processing_date="2026-11-25").to_pylist()[0]
    assert row["video_id"] is None and row["raw_ref"] is None and row["username"] == "a"


@pytest.mark.parametrize("kwargs", [{"severity": "fatal"}, {}])
def test_rejects_bad_input(kwargs) -> None:
    bad = _bad(["1"]) if kwargs else _bad(["1"]).drop_columns(["_errors"])
    with pytest.raises(ValueError):
        quarantine_rows(bad, stage="s", batch_id="b", processing_date="2026-11-25", **kwargs)


def test_write_is_idempotent_per_stage_and_batch(sql_catalog) -> None:
    table = ensure_table(sql_catalog, OPS_QUARANTINE.identifier, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
    common = {"batch_id": "b1", "processing_date": "2026-11-25"}

    assert write_quarantine_rows(table, _bad(["1", "2"]), stage="gold_features", **common) == 2
    assert write_quarantine_rows(table, _bad(["9"]), stage="silver_videos", **common) == 1
    # retry cùng (stage, batch_id): thay đúng phần của nó, không đụng stage khác
    assert write_quarantine_rows(table, _bad(["2"]), stage="gold_features", **common) == 1

    def ids() -> list[tuple[str, str]]:
        t = read_table(table)
        return sorted(zip(t.column("stage").to_pylist(), t.column("video_id").to_pylist(), strict=True))

    assert ids() == [("gold_features", "2"), ("silver_videos", "9")]
    # lần chạy lại không còn lỗi → xoá dòng cũ của lô đó
    assert write_quarantine_rows(table, _bad([]), stage="gold_features", **common) == 0
    assert ids() == [("silver_videos", "9")]


def test_split_valid_output_fits(sql_catalog) -> None:
    t = pa.table({
        "video_id": ["1", "2"], "username": ["u", "u"], "media_type": ["video"] * 2,
        "media_key": ["videos/1/video.mp4", "videos/2/video.mp4"], "thumbnail_key": ["videos/1/cover.jpg", None],
        "posted_at": pa.array([TS] * 2, type=pa.timestamp("us", tz="UTC")),
        "crawl_ts": pa.array([TS] * 2, type=pa.timestamp("us", tz="UTC")), "batch_id": ["b"] * 2,
    })
    res = validate(t, bronze_videos_suite())
    table = ensure_table(sql_catalog, OPS_QUARANTINE.identifier, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
    assert write_quarantine_rows(table, res.bad, stage="bronze_videos", batch_id="b", processing_date=TS) == 1
    assert read_table(table).column("raw_ref").to_pylist() == ["s3://raw/videos/2/"]
