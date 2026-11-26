from __future__ import annotations

from datetime import UTC, date, datetime

import pyarrow as pa
import pytest
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import (
    GOLD_FEATURES,
    GOLD_VIDEO_PREDICTIONS,
    OPS_QUALITY_CHECKS,
    OPS_QUARANTINE,
    SILVER_TRANSCRIPTS,
    SILVER_VIDEOS,
)
from vicap.iceberg.writer import ensure_table, write_table
from vicap.pipelines.embed import to_matrix_column
from vicap.pipelines.quality_check import label_distribution, run_quality_check, total_variation

T0 = datetime(2026, 11, 26, 9, tzinfo=UTC)


def test_label_distribution_and_tv() -> None:
    p = label_distribution([["Sports"], ["Sports", "Tech"]])
    assert p == pytest.approx({"Sports": 2 / 3, "Tech": 1 / 3})
    assert total_variation(p, p) == 0
    assert total_variation({"A": 1.0}, {"B": 1.0}) == 1
    assert label_distribution([]) == {}


def _write(catalog, tdef, rows):
    t = ensure_table(catalog, tdef.identifier, tdef.schema, tdef.partition_spec)
    write_table(t, pa.Table.from_pylist(rows, schema=schema_to_pyarrow(t.schema())))


def test_quality_check(sql_catalog) -> None:
    base = {"username": "a", "media_type": "video", "crawl_ts": T0, "updated_at": T0}
    _write(sql_catalog, SILVER_VIDEOS, [{**base, "video_id": v, "batch_id": b}
                                        for v, b in (("1", "b1"), ("2", "b1"), ("3", "b2"), ("4", "b2"))])
    t = {"transcript_raw": "x", "has_transcript": True, "asr_model": "m", "created_at": T0}
    _write(sql_catalog, SILVER_TRANSCRIPTS, [{**t, "video_id": v, "batch_id": b, "transcript_status": s}
                                             for v, b, s in (("1", "b1", "ok"), ("2", "b1", "ok"),
                                                             ("3", "b2", "ok"), ("4", "b2", "empty_or_music"))])
    import numpy as np

    feats = ensure_table(sql_catalog, GOLD_FEATURES.identifier, GOLD_FEATURES.schema)
    ids = ["1", "2", "3"]  # video 4 chưa có embedding
    write_table(feats, pa.table({
        "video_id": ids, "text_emb": to_matrix_column(np.ones((3, 2, 2), np.float32)),
        "image_emb": to_matrix_column(np.ones((3, 8, 2), np.float32)),
        "n_source_images": pa.array([1] * 3, pa.int32()), "text_encoder": ["t"] * 3, "image_encoder": ["i"] * 3,
        "batch_id": ["b1", "b1", "b2"], "created_at": pa.array([T0] * 3, pa.timestamp("us", tz="UTC")),
    }).cast(schema_to_pyarrow(feats.schema())))
    p = {"username": "a", "probs": [0.1] * 20, "threshold": 0.75, "model_name": "m", "model_version": "1",
         "model_run_id": "r", "predicted_at": T0}
    _write(sql_catalog, GOLD_VIDEO_PREDICTIONS, [{**p, "video_id": v, "batch_id": b, "labels": lb}
                                                 for v, b, lb in (("1", "b1", ["Sports"]), ("2", "b1", ["Sports"]),
                                                                  ("3", "b2", ["Tech"]))])
    _write(sql_catalog, OPS_QUARANTINE, [{"video_id": "4", "username": "a", "stage": "gold_features",
                                          "batch_id": "b2", "processing_date": date(2026, 11, 26),
                                          "errors": ["x"], "severity": "error", "resolved": False,
                                          "created_at": T0}])

    rows = {r.check_name: r for r in run_quality_check("b2", catalog=sql_catalog, now=lambda: T0)}
    assert rows["n_videos"].value == 2 and rows["n_videos"].severity == "ok"
    assert rows["missing_transcript_ratio"].value == 0.5 and rows["missing_transcript_ratio"].severity == "warn"
    assert (rows["missing_embedding"].value, rows["missing_embedding"].detail) == (1, "4")
    assert rows["missing_prediction"].value == 0
    assert rows["quarantined"].detail == '{"gold_features": 1}'
    assert rows["label_shift"].value == 1 and rows["label_shift"].severity == "warn"  # Sports → Tech

    first = {r.check_name: r for r in run_quality_check("b1", catalog=sql_catalog, now=lambda: T0)}
    assert first["label_shift"].value is None and first["n_videos"].value == 2

    run_quality_check("b2", catalog=sql_catalog, now=lambda: T0)  # chạy lại: thay cả lô, không trùng
    out = read_table(sql_catalog.load_table(OPS_QUALITY_CHECKS.identifier))
    assert out.num_rows == 12 and len(set(zip(out.column("batch_id").to_pylist(),
                                              out.column("check_name").to_pylist(), strict=True))) == 12
