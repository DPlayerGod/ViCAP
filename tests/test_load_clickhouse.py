"""load_clickhouse với catalog sqlite + client ClickHouse giả (phần thật: test_integration_clickhouse.py)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.iceberg.tables import GOLD_CREATOR_PROFILE_HISTORY, GOLD_CREATOR_PROFILES, GOLD_VIDEO_PREDICTIONS
from vicap.iceberg.writer import ensure_table, write_table
from vicap.pipelines.load_clickhouse import EPOCH, run_load_clickhouse

T1 = datetime(2026, 11, 26, 9, tzinfo=UTC)
T2 = datetime(2026, 11, 27, 9, tzinfo=UTC)


class _Result:
    def __init__(self, value):
        self.result_rows = [(value,)]


class FakeClickHouse:
    """Giữ các dòng đã insert; ``max(version)`` như ClickHouse (bảng trống → epoch, không có tz)."""

    def __init__(self):
        self.rows: dict[str, list[dict]] = {}
        self.commands: list[str] = []

    def query(self, sql: str):
        col, table = sql.removeprefix("SELECT max(").split(") FROM ")
        vals = [r[col] for r in self.rows.get(table, [])]
        return _Result(max(vals).replace(tzinfo=None) if vals else EPOCH.replace(tzinfo=None))

    def insert_arrow(self, table: str, data: pa.Table):
        self.rows.setdefault(table, []).extend(data.to_pylist())

    def command(self, sql: str):
        self.commands.append(sql)


def _pred(vid: str, ts: datetime) -> dict:
    return {"video_id": vid, "username": "a", "probs": [0.1] * 20, "labels": ["Sports"], "threshold": 0.75,
            "model_name": "kltn-p1", "model_version": "1", "model_run_id": "r", "batch_id": "b",
            "predicted_at": ts}


def _prof(ts: datetime) -> dict:
    return {"username": "a", "n_videos": 1, "label_ratios": [0.1] * 19, "main_label": "Sports",
            "confidence": "low", "model_run_id": "r", "batch_id": "b", "updated_at": ts}


def _write(table, rows):
    write_table(table, pa.Table.from_pylist(rows, schema=schema_to_pyarrow(table.schema())))


def test_incremental_by_watermark(sql_catalog) -> None:
    preds = ensure_table(sql_catalog, GOLD_VIDEO_PREDICTIONS.identifier, GOLD_VIDEO_PREDICTIONS.schema)
    prof = ensure_table(sql_catalog, GOLD_CREATOR_PROFILES.identifier, GOLD_CREATOR_PROFILES.schema)
    _write(preds, [_pred("1", T1), _pred("2", T1)])
    _write(prof, [_prof(T1)])
    ch = FakeClickHouse()

    def run(**kw):
        return run_load_clickhouse(client=ch, catalog=sql_catalog, database="vicap", **kw).loaded

    # history chưa có trên Iceberg → bỏ qua
    assert run() == {"video_predictions": 2, "creator_profiles": 1, "creator_profile_history": 0}
    row = ch.rows["vicap.video_predictions"][0]
    assert list(row) == [f.name for f in GOLD_VIDEO_PREDICTIONS.schema.fields]
    assert run() == {"video_predictions": 0, "creator_profiles": 0, "creator_profile_history": 0}

    _write(preds, [_pred("3", T2)])
    hist = ensure_table(sql_catalog, GOLD_CREATOR_PROFILE_HISTORY.identifier, GOLD_CREATOR_PROFILE_HISTORY.schema,
                        GOLD_CREATOR_PROFILE_HISTORY.partition_spec)
    _write(hist, [{"snapshot_date": date(2026, 11, 27), **_prof(T2)}])
    assert run() == {"video_predictions": 1, "creator_profiles": 0, "creator_profile_history": 1}
    assert [r["video_id"] for r in ch.rows["vicap.video_predictions"]] == ["1", "2", "3"]

    assert run(full=True)["video_predictions"] == 3
    assert "OPTIMIZE TABLE vicap.video_predictions FINAL" in ch.commands
