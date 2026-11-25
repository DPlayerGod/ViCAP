"""Test các check (PyArrow + NumPy): cấp lô qua run_check / run_suite, cấp dòng qua row_mask / split_valid."""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow as pa
import pytest

from vicap.quality import (
    ColumnRelation,
    Covers,
    Dimension,
    ForeignKey,
    InRange,
    InSet,
    ListLength,
    MatchesDate,
    MeanShift,
    NonEmpty,
    NonZeroVector,
    NotNull,
    Positive,
    RequiredColumns,
    Severity,
    Unique,
    VectorDtype,
    VectorRange,
    VectorShape,
    row_mask,
    run_check,
    run_suite,
    split_valid,
)


def T(**cols) -> pa.Table:
    return pa.table(cols)


def matrix(rows: int, cols: int, value: float = 1.0) -> list[list[float]]:
    return [[value] * cols for _ in range(rows)]


# ─────────────────────────── COMPLETENESS ──────────────────────────────────


def test_required_columns_pass_and_fail() -> None:
    t = T(a=[1], b=[2])
    assert run_check(t, RequiredColumns(("a", "b"))).passed
    out = run_check(t, RequiredColumns(("a", "c")))
    assert not out.passed and out.dimension is Dimension.COMPLETENESS and "c" in (out.message or "")


def test_not_null_detects_nulls_and_missing_column() -> None:
    t = T(a=[1, None], b=[1, 2])
    assert not run_check(t, NotNull(("a",))).passed
    assert run_check(t, NotNull(("b",))).passed
    assert not run_check(t, NotNull(("missing",))).passed
    assert row_mask(t, NotNull(("a", "b"))).tolist() == [True, False]


def test_covers_reports_missing_reference_keys() -> None:
    silver = T(video_id=["1", "2", "3"])
    out = run_check(T(video_id=["1", "3"]), Covers("video_id", silver, "video_id"))
    assert not out.passed and out.failing_rows == 1 and out.severity is Severity.WARN


# ──────────────────────────── UNIQUENESS ───────────────────────────────────


def test_unique_flags_duplicate_composite_key() -> None:
    t = T(s=["A", "A", "B"], d=[1, 1, 1])
    out = run_check(t, Unique(("s", "d")))
    assert not out.passed and out.failing_rows == 1
    assert run_check(T(s=["A", "B"]), Unique(("s",))).passed


# ───────────────────────────── VALIDITY ────────────────────────────────────


def test_in_range_ignores_null_flags_out_of_bounds() -> None:
    t = T(x=[10.0, 150.0, None])
    out = run_check(t, InRange("x", min_value=0, max_value=100))
    assert not out.passed and out.failing_rows == 1
    assert row_mask(t, InRange("x", 0, 100)).tolist() == [True, False, True]


def test_positive_flags_non_positive_only_on_non_null() -> None:
    t = T(p=[1.0, 0.0, None], q=[5.0, 6.0, 7.0])
    assert not run_check(t, Positive(("p",))).passed
    assert run_check(t, Positive(("q",))).passed


def test_in_set_enforces_allowed_values() -> None:
    t = T(media_type=["video", "reel", None])
    out = run_check(t, InSet("media_type", ("video", "slideshow")))
    assert not out.passed and out.failing_rows == 1


def test_list_length() -> None:
    t = T(labels=[["Food"], [], None])
    assert row_mask(t, ListLength("labels", min_len=1)).tolist() == [True, False, True]


def test_vector_dtype() -> None:
    t = T(emb=[matrix(2, 3)], probs=[[0.1, 0.2]], ints=[[[1]]])
    assert run_check(t, VectorDtype("emb", depth=2)).passed
    assert not run_check(t, VectorDtype("probs", depth=2)).passed
    assert not run_check(t, VectorDtype("ints", depth=2)).passed


def test_vector_shape_2d() -> None:
    good = matrix(2, 3)
    nan = matrix(2, 3)
    nan[1][2] = float("nan")
    t = T(emb=[good, matrix(1, 3), [[1.0, 1.0, 1.0], [1.0, 1.0]], nan, None, [[1.0, None, 1.0], [1.0] * 3]])
    assert row_mask(t, VectorShape("emb", (2, 3))).tolist() == [True, False, False, False, False, False]


def test_vector_shape_1d_and_fixed_size_list() -> None:
    t = T(p=[[0.1, 0.2, 0.3], [0.1, 0.2]])
    assert row_mask(t, VectorShape("p", (3,))).tolist() == [True, False]
    fixed = pa.FixedSizeListArray.from_arrays(
        pa.FixedSizeListArray.from_arrays(pa.array(np.ones(12, dtype=np.float32)), 3), 2
    )
    assert row_mask(pa.table({"emb": fixed}), VectorShape("emb", (2, 3))).tolist() == [True, True]


def test_non_zero_vector_2d_all_rows_and_selected_rows() -> None:
    zero_second = [[1.0, 2.0], [0.0, 0.0]]
    zero_first = [[0.0, 0.0], [1.0, 2.0]]
    t = T(emb=[matrix(2, 2), zero_second, zero_first, None])
    assert row_mask(t, NonZeroVector("emb")).tolist() == [True, False, False, False]
    assert row_mask(t, NonZeroVector("emb", rows=(0,))).tolist() == [True, True, False, False]


def test_non_zero_vector_1d() -> None:
    assert row_mask(T(v=[[0.0, 0.0], [0.0, 1.0]]), NonZeroVector("v")).tolist() == [False, True]


def test_vector_range() -> None:
    t = T(p=[[0.0, 1.0], [0.5, 1.2], [float("nan"), 0.1], None])
    assert row_mask(t, VectorRange("p", 0, 1)).tolist() == [True, False, False, True]


# ──────────────────────────── CONSISTENCY ──────────────────────────────────


def test_column_relation_cross_field() -> None:
    t = T(posted=[1, 5, None], crawled=[2, 3, 4])
    out = run_check(t, ColumnRelation("posted", "<=", "crawled"))
    assert not out.passed and out.failing_rows == 1 and out.dimension is Dimension.CONSISTENCY


def test_column_relation_rejects_unknown_operator() -> None:
    assert not run_check(T(a=[1], b=[2]), ColumnRelation("a", "<>", "b")).passed


def test_foreign_key_detects_orphans_and_skips_empty() -> None:
    ref = T(key=[1, 2, 3])
    out = run_check(T(key=[1, 99]), ForeignKey("key", ref, "key"))
    assert not out.passed and out.failing_rows == 1
    empty = pa.table({"key": pa.array([], type=pa.int64())})
    assert run_check(empty, ForeignKey("key", ref, "key")).passed


# ───────────────────────────── ACCURACY ────────────────────────────────────


def test_mean_shift_numeric_and_vector() -> None:
    prev = T(has=[True, True, False, False], probs=[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
    same = T(has=[True, False], probs=[[0.1, 0.9], [0.1, 0.9]])
    shifted = T(has=[True, True], probs=[[0.9, 0.1], [0.9, 0.1]])
    assert run_check(same, MeanShift("has", prev, 0.2)).passed
    out = run_check(shifted, MeanShift("has", prev, 0.2))
    assert not out.passed and out.severity is Severity.WARN and out.dimension is Dimension.ACCURACY
    assert run_check(same, MeanShift("probs", prev, 0.1)).passed
    assert not run_check(shifted, MeanShift("probs", prev, 0.1)).passed
    assert run_check(shifted, MeanShift("probs", None, 0.1)).passed  # chưa có lô trước


# ──────────────────────────── TIMELINESS ───────────────────────────────────


def test_non_empty() -> None:
    assert run_check(T(a=[1]), NonEmpty()).passed
    out = run_check(pa.table({"a": pa.array([], type=pa.int64())}), NonEmpty())
    assert not out.passed and out.dimension is Dimension.TIMELINESS


def test_matches_date_flags_rows_outside_target_day() -> None:
    t = T(d=[date(2026, 6, 15), date(2026, 6, 16)])
    out = run_check(t, MatchesDate("d", "2026-06-15"))
    assert not out.passed and out.failing_rows == 1


# ─────────────────────────── engine ────────────────────────────────────────


def test_run_suite_separates_errors_from_warnings() -> None:
    t = T(x=[1, None], posted=[5, 1], crawled=[1, 2])
    res = run_suite(t, [NotNull(("x",)), ColumnRelation("posted", "<=", "crawled", severity=Severity.WARN)], "demo")
    assert res.is_valid is False and len(res.errors) == 1 and len(res.warnings) == 1
    assert all(msg.startswith("demo: ") for msg in res.errors)


def test_split_valid_collects_error_labels() -> None:
    t = T(id=["a", "b", None], v=[[1.0], [0.0], [1.0]])
    ok, bad = split_valid(t, [NotNull(("id",)), NonZeroVector("v")])
    assert ok.column("id").to_pylist() == ["a"]
    assert bad.column("id").to_pylist() == ["b", None]
    assert bad.column("_errors").to_pylist() == [["non_zero_vector:v"], ["not_null:id"]]


def test_split_valid_no_checks_and_empty_table() -> None:
    t = T(id=["a"])
    ok, bad = split_valid(t, [])
    assert ok.num_rows == 1 and bad.num_rows == 0 and "_errors" in bad.column_names
    empty = pa.table({"v": pa.array([], type=pa.list_(pa.list_(pa.float32())))})
    ok, bad = split_valid(empty, [VectorShape("v", (2, 3)), NonZeroVector("v", rows=(0,))])
    assert ok.num_rows == 0 and bad.num_rows == 0


def test_row_mask_rejects_batch_only_check_and_missing_column() -> None:
    with pytest.raises(TypeError):
        row_mask(T(a=[1]), Unique(("a",)))
    with pytest.raises(ValueError, match="missing"):
        row_mask(T(a=[1]), NotNull(("b",)))
