from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa
from pyiceberg.io.pyarrow import schema_to_pyarrow

from vicap.iceberg.reader import read_table
from vicap.iceberg.tables import GOLD_FEATURES, GOLD_VIDEO_PREDICTIONS, SILVER_VIDEOS
from vicap.iceberg.writer import ensure_table, write_table
from vicap.model.predict import P1, predict_batch
from vicap.model.register_p1 import checkpoint_params
from vicap.pipelines.embed import to_matrix_column
from vicap.pipelines.predict import run_predict

T0 = datetime(2026, 11, 26, 9, tzinfo=UTC)
LABELS = [f"L{i}" for i in range(20)]


class FakePredictor:
    labels, threshold = LABELS, 0.75

    def __init__(self):
        self.inputs = []

    def predict(self, xs):
        xt, xv = xs
        self.inputs.append((xt.shape, xv.shape))
        p = np.full((len(xt), 20), 0.1, np.float32)
        p[:, 0] = xt[:, 0]  # video 1: 0,9 → L0; video 2: 0,2 → không qua ngưỡng
        p[:, 3] = 0.8
        return p

    def decide(self, P):
        D = P >= self.threshold
        D[np.arange(len(P)), P.argmax(1)] = True
        return [[self.labels[j] for j in np.where(r)[0]] for r in D]


def fake_p1(version: str = "1") -> P1:
    return P1(FakePredictor(), "kltn-p1", version, f"run{version}")


def test_predict_batch_shapes_and_decide() -> None:
    te = np.zeros((2, 2, 768), np.float32)
    te[:, 0, 0] = [0.9, 0.2]
    ie = np.ones((2, 8, 768), np.float32)
    p1 = fake_p1()
    probs, labels = predict_batch(p1, te, ie)
    assert p1.predictor.inputs == [((2, 1536), (2, 768))]
    assert probs.shape == (2, 20) and labels == [["L0", "L3"], ["L3"]]
    assert predict_batch(p1, te[:0], ie[:0])[1] == []


def test_checkpoint_params() -> None:
    ck = {"arch": {"type": "TwoTower", "dims": [1536, 768], "d": 256}, "norm": "zscore", "threshold": 0.75,
          "labels": LABELS, "n_train": 1900}
    p = checkpoint_params(ck)
    assert (p["arch"], p["dims"], p["threshold"], p["n_labels"], p["n_train"]) == ("TwoTower", [1536, 768], 0.75,
                                                                                     20, 1900)


def test_run_predict(sql_catalog) -> None:
    silver = ensure_table(sql_catalog, SILVER_VIDEOS.identifier, SILVER_VIDEOS.schema)
    feats = ensure_table(sql_catalog, GOLD_FEATURES.identifier, GOLD_FEATURES.schema)
    base = {"media_type": "video", "crawl_ts": T0, "batch_id": "b1", "updated_at": T0}
    write_table(silver, pa.Table.from_pylist([{**base, "video_id": "1", "username": "a"},
                                              {**base, "video_id": "2", "username": "b"}],
                                             schema=schema_to_pyarrow(silver.schema())))
    te = np.zeros((2, 2, 768), np.float32)
    te[:, 0, 0] = [0.9, 0.2]
    rows = pa.table({
        "video_id": ["1", "2"], "text_emb": to_matrix_column(te),
        "image_emb": to_matrix_column(np.ones((2, 8, 768), np.float32)),
        "n_source_images": pa.array([1, 1], pa.int32()),
        "text_encoder": ["t", "t"], "image_encoder": ["i", "i"], "batch_id": ["b1", "b1"],
        "created_at": pa.array([T0, T0], pa.timestamp("us", tz="UTC")),
    }).cast(schema_to_pyarrow(feats.schema()))
    write_table(feats, rows)

    s = run_predict(catalog=sql_catalog, p1=fake_p1(), now=lambda: T0)
    assert (s.written, s.quarantined, s.skipped) == (2, 0, 0)
    out = {r["video_id"]: r for r in read_table(sql_catalog.load_table(GOLD_VIDEO_PREDICTIONS.identifier)).to_pylist()}
    assert out["1"]["labels"] == ["L0", "L3"] and out["2"]["labels"] == ["L3"]
    assert (out["1"]["username"], out["1"]["model_version"], out["1"]["model_run_id"]) == ("a", "1", "run1")
    assert len(out["1"]["probs"]) == 20 and out["1"]["threshold"] == 0.75

    assert run_predict(catalog=sql_catalog, p1=fake_p1(), now=lambda: T0).skipped == 2  # cùng phiên bản
    assert run_predict(catalog=sql_catalog, p1=fake_p1("2"), now=lambda: T0).written == 2  # đổi alias → chạy lại
