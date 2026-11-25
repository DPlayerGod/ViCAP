"""Suite theo bảng (PLAN mục 4.1, 9.3): luật embedding và hai cấp kiểm tra."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pyarrow as pa

from vicap.quality import (
    bronze_videos_suite,
    gold_features_suite,
    gold_predictions_suite,
    gold_profiles_suite,
    validate,
)

UTC = UTC


def emb(rows: int, dim: int = 768, value: float = 0.5) -> np.ndarray:
    return np.full((rows, dim), value, dtype=np.float32)


def features_table(n: int = 5, **overrides) -> pa.Table:
    text = [emb(2) for _ in range(n)]
    image = [emb(8) for _ in range(n)]
    for i, (kind, arr) in overrides.items():
        (text if kind == "text" else image)[int(i)] = arr
    to_list = pa.list_(pa.list_(pa.float32()))
    return pa.table({
        "video_id": [str(1000 + i) for i in range(n)],
        "text_emb": pa.array([a.tolist() for a in text], type=to_list),
        "image_emb": pa.array([a.tolist() for a in image], type=to_list),
        "text_encoder": ["vinai/phobert-base"] * n,
        "image_encoder": ["google/siglip2-base-patch16-224"] * n,
        "batch_id": ["20261125-000000"] * n,
    })


def test_features_all_valid() -> None:
    res = validate(features_table(), gold_features_suite())
    assert res.batch.is_valid and res.ok.num_rows == 5 and res.bad.num_rows == 0


def test_features_transcript_vector_may_be_zero() -> None:
    no_speech = emb(2)
    no_speech[1] = 0
    res = validate(features_table(**{"0": ("text", no_speech)}), gold_features_suite())
    assert res.ok.num_rows == 5


def test_features_row_errors_are_isolated() -> None:
    caption_zero = emb(2)
    caption_zero[0] = 0
    nan_image = emb(8)
    nan_image[3, 10] = np.nan
    res = validate(
        features_table(**{
            "0": ("text", caption_zero),
            "1": ("image", emb(7)),  # thiếu 1 ảnh
            "2": ("image", nan_image),
            "3": ("text", emb(2, dim=512)),
        }),
        gold_features_suite(),
    )
    assert res.batch.is_valid
    assert res.ok.column("video_id").to_pylist() == ["1004"]
    errors = dict(zip(res.bad.column("video_id").to_pylist(), res.bad.column("_errors").to_pylist(), strict=True))
    assert errors["1000"] == ["non_zero_vector:text_emb[0]"]
    assert errors["1001"] == ["vector_shape:image_emb"]
    assert errors["1002"] == ["vector_shape:image_emb"]
    assert "vector_shape:text_emb" in errors["1003"]


def test_features_one_zero_image_is_isolated() -> None:
    zero_frame = emb(8)
    zero_frame[5] = 0
    res = validate(features_table(**{"4": ("image", zero_frame)}), gold_features_suite())
    assert res.ok.num_rows == 4
    assert res.bad.column("_errors").to_pylist() == [["non_zero_vector:image_emb"]]


def test_features_batch_errors_stop_batch() -> None:
    t = features_table(2)
    dup = pa.concat_tables([t, t])
    res = validate(dup, gold_features_suite())
    assert not res.batch.is_valid and res.ok.num_rows == 0 and res.bad.num_rows == 4

    flat = t.set_column(1, "text_emb", pa.array([[0.5] * 1536] * 2, type=pa.list_(pa.float32())))
    res = validate(flat, gold_features_suite())
    assert not res.batch.is_valid and any("text_emb has type" in e for e in res.batch.errors)

    res = validate(t.drop_columns(["image_emb"]), gold_features_suite())
    assert not res.batch.is_valid and "missing required columns" in res.batch.errors[0]


def test_features_coverage_is_warning() -> None:
    silver = pa.table({"video_id": ["1000", "1001", "9999"]})
    res = validate(features_table(2), gold_features_suite(silver=silver))
    assert res.batch.is_valid and len(res.batch.warnings) == 1


def _bronze(n: int) -> pa.Table:
    ts = datetime(2026, 11, 25, tzinfo=UTC)
    return pa.table({
        "video_id": [str(i) for i in range(n)],
        "username": ["u"] * n,
        "media_type": ["video"] * n,
        "media_key": [f"videos/{i}.mp4" for i in range(n)],
        "thumbnail_key": [f"thumbnails/{i}.jpg" for i in range(n)],
        "posted_at": pa.array([ts] * n, type=pa.timestamp("us", tz="UTC")),
        "crawl_ts": pa.array([ts] * n, type=pa.timestamp("us", tz="UTC")),
        "batch_id": ["b"] * n,
    })


def test_bronze_batch_of_five_with_one_missing_thumbnail() -> None:
    t = _bronze(5)
    t = t.set_column(4, "thumbnail_key", pa.array(["thumbnails/0.jpg", None, "thumbnails/2.jpg",
                                                   "thumbnails/3.jpg", "thumbnails/4.jpg"]))
    res = validate(t, bronze_videos_suite())
    assert res.batch.is_valid and res.ok.num_rows == 4 and res.bad.num_rows == 1
    assert res.bad.column("_errors").to_pylist() == [["not_null:video_id,username,media_key,thumbnail_key"]]


def test_bronze_empty_batch_stops() -> None:
    res = validate(_bronze(0), bronze_videos_suite())
    assert not res.batch.is_valid and "batch is empty" in res.batch.errors[0]


def test_predictions_suite() -> None:
    probs = [[0.1] * 20, [0.1] * 19 + [1.5], [0.1] * 19]
    t = pa.table({
        "video_id": ["1", "2", "3"],
        "username": ["u"] * 3,
        "probs": pa.array(probs, type=pa.list_(pa.float32())),
        "labels": [["Food"], ["Food"], ["Food"]],
        "threshold": [0.75] * 3,
        "model_run_id": ["r"] * 3,
        "batch_id": ["b"] * 3,
    })
    features = pa.table({"video_id": ["1", "2"]})
    res = validate(t, gold_predictions_suite(features=features))
    assert res.ok.column("video_id").to_pylist() == ["1"]
    errors = dict(zip(res.bad.column("video_id").to_pylist(), res.bad.column("_errors").to_pylist(), strict=True))
    assert errors["2"] == ["vector_range:probs"]
    assert errors["3"] == ["vector_shape:probs", "foreign_key:video_id"]


def test_profiles_suite() -> None:
    t = pa.table({
        "username": ["a", "b"],
        "n_videos": [3, 12],
        "label_ratios": pa.array([[0.05] * 19, [0.05] * 19], type=pa.list_(pa.float32())),
        "confidence": ["low", "very-high"],
        "batch_id": ["b", "b"],
    })
    res = validate(t, gold_profiles_suite())
    assert res.ok.column("username").to_pylist() == ["a"]
    assert res.bad.column("_errors").to_pylist() == [["in_set:confidence"]]
