"""load_clickhouse trên ClickHouse + Iceberg REST thật: database ClickHouse tạm dựng từ infra/clickhouse/init.sql
và namespace Iceberg tạm. Cần ``make up-serving`` (hoặc profile core)."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import pyarrow as pa
import pytest
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.iceberg.tables import GOLD_CREATOR_PROFILE_HISTORY, GOLD_CREATOR_PROFILES, GOLD_VIDEO_PREDICTIONS
from vicap.iceberg.writer import replace_by_keys, write_table
from vicap.pipelines.load_clickhouse import run_load_clickhouse

pytestmark = pytest.mark.integration

INIT_SQL = Path(__file__).resolve().parents[1] / "infra" / "clickhouse" / "init.sql"
T1 = datetime(2026, 11, 26, 9, tzinfo=UTC)
T2 = datetime(2026, 11, 27, 9, tzinfo=UTC)


@pytest.fixture
def ch_db():
    from vicap.clickhouse.client import get_clickhouse_client

    try:
        client = get_clickhouse_client()
        client.command("SELECT 1")
    except Exception:  # noqa: BLE001
        pytest.skip("ClickHouse chưa chạy — make up-serving")
    db = f"it_{uuid4().hex[:8]}"
    sql = re.sub(r"--[^\n]*", "", INIT_SQL.read_text(encoding="utf-8")).replace("vicap.", f"{db}.")
    sql = sql.replace("DATABASE IF NOT EXISTS vicap", f"DATABASE IF NOT EXISTS {db}")
    for stmt in filter(str.strip, sql.split(";")):
        client.command(stmt)
    try:
        yield client, db
    finally:
        client.command(f"DROP DATABASE IF EXISTS {db}")


def _rows(table, rows):
    return pa.Table.from_pylist(rows, schema=schema_to_pyarrow(table.schema()))


def test_load_real_clickhouse(ch_db, iceberg_scratch) -> None:
    client, db = ch_db
    preds = iceberg_scratch("video_predictions", GOLD_VIDEO_PREDICTIONS)
    prof = iceberg_scratch("creator_profiles", GOLD_CREATOR_PROFILES)
    hist = iceberg_scratch("creator_profile_history", GOLD_CREATOR_PROFILE_HISTORY)
    ns = preds.name()[-2]
    sources = {t: f"{ns}.{t}" for t in ("video_predictions", "creator_profiles", "creator_profile_history")}

    def pred(vid, ts, label):
        return {"video_id": vid, "username": "a", "probs": [0.5] * 20, "labels": [label], "threshold": 0.75,
                "model_name": "kltn-p1", "model_version": "1", "model_run_id": "r", "batch_id": "b",
                "predicted_at": ts}

    prof_row = {"username": "a", "n_videos": 2, "label_ratios": [0.25] * 19, "main_label": None,
                "confidence": "low", "model_run_id": "r", "batch_id": "b", "updated_at": T1}
    write_table(preds, _rows(preds, [pred("1", T1, "Sports"), pred("2", T1, "Tech")]))
    write_table(prof, _rows(prof, [prof_row]))
    write_table(hist, _rows(hist, [{"snapshot_date": date(2026, 11, 26), **prof_row}]))

    def run(**kw):
        return run_load_clickhouse(client=client, database=db, sources=sources, **kw).loaded

    assert run() == {"video_predictions": 2, "creator_profiles": 1, "creator_profile_history": 1}
    assert set(run().values()) == {0}

    replace_by_keys(preds, _rows(preds, [pred("1", T2, "Food_Review")]))  # dự đoán lại video 1
    assert run()["video_predictions"] == 1
    final = dict(client.query(f"SELECT video_id, labels FROM {db}.video_predictions FINAL ORDER BY video_id")
                 .result_rows)
    assert final == {"1": ["Food_Review"], "2": ["Tech"]}
    p = client.query(f"SELECT n_videos, length(label_ratios), main_label, updated_at FROM {db}.creator_profiles "
                     "FINAL").result_rows
    assert p[0][:3] == (2, 19, None) and p[0][3].replace(tzinfo=UTC) == T1

    assert run(full=True)["video_predictions"] == 2
    assert client.query(f"SELECT count() FROM {db}.video_predictions").result_rows[0][0] == 2  # sau OPTIMIZE FINAL
