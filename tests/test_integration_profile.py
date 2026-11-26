"""profile trên Spark + Iceberg REST thật, trong namespace tạm: ``make test-spark``. Host không có pyspark → bỏ qua."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

pytest.importorskip("pyspark")

import pyarrow as pa  # noqa: E402
from pyiceberg.io.pyarrow import schema_to_pyarrow  # noqa: E402

from vicap.config import ModelConfig  # noqa: E402
from vicap.iceberg.reader import read_table  # noqa: E402
from vicap.iceberg.tables import (  # noqa: E402
    GOLD_CREATOR_PROFILE_HISTORY,
    GOLD_CREATOR_PROFILES,
    GOLD_VIDEO_PREDICTIONS,
    OPS_QUARANTINE,
)
from vicap.iceberg.writer import replace_by_keys  # noqa: E402
from vicap.pipelines.profile import run_profile  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.spark]

T0 = datetime(2026, 11, 26, 9, tzinfo=UTC)
LABELS = ModelConfig().labels
OTHERS = LABELS.index("Others")


@pytest.fixture(scope="module")
def spark():
    from vicap.spark import get_spark

    s = get_spark("test_profile")
    yield s
    s.stop()


def _probs(hot: str, p: float = 0.9) -> list[float]:
    v = [0.05] * len(LABELS)
    v[LABELS.index(hot)] = p
    v[OTHERS] = 0.99  # Others cao nhất nhưng bị bỏ khỏi hồ sơ
    return v


def _preds(table, rows, hour: int = 9):
    base = {"threshold": 0.75, "model_name": "kltn-p1", "model_version": "1", "labels": ["x"]}
    return pa.Table.from_pylist(
        [{**base, "video_id": v, "username": u, "probs": p, "batch_id": b, "model_run_id": r,
          "predicted_at": T0.replace(hour=hour, minute=i)} for i, (v, u, p, b, r) in enumerate(rows)],
        schema=schema_to_pyarrow(table.schema()))


def test_profile_mean_merge_history(spark, iceberg_scratch) -> None:
    preds = iceberg_scratch("video_predictions", GOLD_VIDEO_PREDICTIONS)
    prof = iceberg_scratch("creator_profiles", GOLD_CREATOR_PROFILES)
    hist = iceberg_scratch("creator_profile_history", GOLD_CREATOR_PROFILE_HISTORY)
    iceberg_scratch("quarantine", OPS_QUARANTINE)
    ns = preds.name()[-2]
    names = {k: f"{ns}.{v}" for k, v in (("predictions_table", "video_predictions"),
                                         ("profiles_table", "creator_profiles"),
                                         ("history_table", "creator_profile_history"),
                                         ("quarantine_table", "quarantine"))}

    def run(*batches: str, day: date = date(2026, 11, 26)):
        return run_profile(list(batches) or None, spark=spark, processing_date=day, now=lambda: T0, **names)

    rows = [(str(i), "a", _probs("Sports"), "b1", "r1") for i in range(4)]
    rows += [("10", "b", _probs("Tech", 0.6), "b1", "r1"), ("11", "b", _probs("Food_Review", 0.8), "b1", "r1")]
    replace_by_keys(preds, _preds(preds, rows))

    s = run()
    assert (s.videos, s.creators, s.inserted, s.history, s.quarantined) == (6, 2, 2, 2, 0)
    out = {r["username"]: r for r in read_table(prof.refresh()).to_pylist()}
    keep = [j for j in range(len(LABELS)) if j != OTHERS]
    want_b = np.mean([np.array(_probs("Tech", 0.6))[keep], np.array(_probs("Food_Review", 0.8))[keep]], 0)
    assert np.allclose(out["b"]["label_ratios"], want_b, atol=1e-6) and len(out["b"]["label_ratios"]) == 19
    assert (out["a"]["main_label"], out["a"]["n_videos"], out["a"]["confidence"]) == ("Sports", 4, "low")
    assert out["b"]["main_label"] == "Food_Review"

    # chạy lại: không đổi gì, không snapshot mới
    snaps = (len(prof.refresh().snapshots()), len(hist.refresh().snapshots()))
    assert run().unchanged == 2
    assert (len(prof.refresh().snapshots()), len(hist.refresh().snapshots())) == snaps

    # lô b2 có thêm video của a: chỉ a được tính lại, history có ảnh chụp ngày mới
    replace_by_keys(preds, _preds(preds, [("5", "a", _probs("Sports"), "b2", "r2")], hour=10))
    s = run("b2", day=date(2026, 11, 27))
    assert (s.creators, s.updated, s.history) == (1, 1, 1)
    a = next(r for r in read_table(prof.refresh()).to_pylist() if r["username"] == "a")
    assert (a["n_videos"], a["confidence"], a["batch_id"], a["model_run_id"]) == (5, "medium", "b2", "r2")
    h = sorted((r["snapshot_date"].isoformat(), r["username"], r["n_videos"])
               for r in read_table(hist.refresh()).to_pylist())
    assert h == [("2026-11-26", "a", 4), ("2026-11-26", "b", 2), ("2026-11-27", "a", 5)]
