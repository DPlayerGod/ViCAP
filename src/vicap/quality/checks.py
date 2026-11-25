"""Các check chất lượng dữ liệu tái sử dụng, viết bằng PyArrow + NumPy.

Mỗi check là một cặp *config* (dataclass nhỏ: cột nào, ngưỡng bao nhiêu) + hàm
``check(table, config) -> CheckOutcome``. Cùng một hàm dùng lại cho mọi bảng, chỉ khác config.
Dữ liệu vào là ``pa.Table`` (PyIceberg đọc / ghi Arrow; Spark 3.5 chuyển được sang Arrow).

| Chiều         | Config                                                                 |
|---------------|------------------------------------------------------------------------|
| COMPLETENESS  | ``RequiredColumns`` · ``NotNull`` · ``Covers``                         |
| UNIQUENESS    | ``Unique``                                                             |
| VALIDITY      | ``InRange`` · ``Positive`` · ``InSet`` · ``ListLength``                |
|               | ``VectorDtype`` · ``VectorShape`` · ``NonZeroVector`` · ``VectorRange``|
| CONSISTENCY   | ``ColumnRelation`` · ``ForeignKey``                                    |
| ACCURACY      | ``MeanShift`` (so với lô trước, mặc định WARN)                         |
| TIMELINESS    | ``NonEmpty`` · ``MatchesDate``                                         |

Hai cấp (PLAN mục 9.3):

- **Cấp lô:** ``run_suite(table, checks)`` → ``ValidationResult`` (ERROR thì dừng lô, WARN thì ghi nhận).
- **Cấp dòng:** các check có trong ``ROW_CHECKS`` còn có ``row_mask(table, config) -> np.ndarray[bool]``
  (True = dòng hợp lệ). ``split_valid(table, row_checks) -> (ok, bad)`` tách dòng lỗi; ``bad`` có cột
  ``_errors`` ghi tên các check bị vi phạm.

Quy ước null: các check giá trị (``InRange``, ``InSet``, ``ForeignKey``, ``VectorRange``, …) bỏ qua null,
null do ``NotNull`` bắt. Riêng ``VectorShape`` / ``NonZeroVector`` coi null là lỗi, vì vector phải có.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from vicap.quality.result import CheckOutcome, Dimension, Severity, ValidationResult

# ───────────────────────── Config objects (1 dataclass / 1 loại check) ──────


@dataclass(frozen=True)
class RequiredColumns:
    """COMPLETENESS — các cột bắt buộc phải tồn tại."""

    columns: tuple[str, ...]
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class NotNull:
    """COMPLETENESS — các cột không được null."""

    columns: tuple[str, ...]
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class Covers:
    """COMPLETENESS — mọi khoá của ``reference`` phải có trong ``column`` (vd mọi video Silver có embedding)."""

    column: str
    reference: pa.Table
    reference_column: str
    severity: Severity = Severity.WARN


@dataclass(frozen=True)
class Unique:
    """UNIQUENESS — tổ hợp ``columns`` phải duy nhất."""

    columns: tuple[str, ...]
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class InRange:
    """VALIDITY — giá trị nằm trong ``[min_value, max_value]``."""

    column: str
    min_value: float | None = None
    max_value: float | None = None
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class Positive:
    """VALIDITY — các cột phải > 0."""

    columns: tuple[str, ...]
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class InSet:
    """VALIDITY — giá trị thuộc tập cho phép."""

    column: str
    allowed: tuple[object, ...]
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class ListLength:
    """VALIDITY — độ dài list trong ``[min_len, max_len]`` (vd ``labels`` có ít nhất 1 nhãn)."""

    column: str
    min_len: int = 0
    max_len: int | None = None
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class VectorDtype:
    """VALIDITY (cấp cột) — cột là list lồng ``depth`` cấp của số thực (vd embedding 2×768: depth=2)."""

    column: str
    depth: int
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class VectorShape:
    """VALIDITY — mỗi dòng đúng ``shape`` (1 hoặc 2 chiều), không NaN / null bên trong; dòng null là lỗi."""

    column: str
    shape: tuple[int, ...]
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class NonZeroVector:
    """VALIDITY — vector khác 0.

    Với cột 2 chiều: ``rows=None`` đòi mọi vector con khác 0; ``rows=(0,)`` chỉ đòi vector con số 0
    (vd ``text_emb[0]`` phải khác 0, ``text_emb[1]`` được phép bằng 0).
    """

    column: str
    rows: tuple[int, ...] | None = None
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class VectorRange:
    """VALIDITY — mọi phần tử của vector trong ``[min_value, max_value]`` (vd xác suất trong [0, 1])."""

    column: str
    min_value: float
    max_value: float
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class ColumnRelation:
    """CONSISTENCY — quan hệ giữa 2 cột trong cùng dòng, ``op`` ∈ {">=", "<=", ">", "<", "==", "!="}."""

    left: str
    op: str
    right: str
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class ForeignKey:
    """CONSISTENCY — mọi giá trị ``column`` phải có trong ``reference[reference_column]``."""

    column: str
    reference: pa.Table
    reference_column: str
    name: str | None = None
    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class MeanShift:
    """ACCURACY — trung bình của cột (số / bool, hoặc từng vị trí của vector 1 chiều) không lệch quá
    ``max_abs_diff`` so với ``reference`` (lô trước). Bỏ qua khi chưa có lô trước."""

    column: str
    reference: pa.Table | None
    max_abs_diff: float
    severity: Severity = Severity.WARN


@dataclass(frozen=True)
class NonEmpty:
    """TIMELINESS — lô không rỗng."""

    severity: Severity = Severity.ERROR


@dataclass(frozen=True)
class MatchesDate:
    """TIMELINESS — mọi dòng đúng ngày kỳ vọng (vd ``processing_date == ds``)."""

    column: str
    expected: str
    severity: Severity = Severity.ERROR


Check = (
    RequiredColumns
    | NotNull
    | Covers
    | Unique
    | InRange
    | Positive
    | InSet
    | ListLength
    | VectorDtype
    | VectorShape
    | NonZeroVector
    | VectorRange
    | ColumnRelation
    | ForeignKey
    | MeanShift
    | NonEmpty
    | MatchesDate
)


# ─────────────────────────────── Helpers ────────────────────────────────────


def _missing_columns(table: pa.Table, columns: Iterable[str]) -> list[str]:
    return sorted(set(columns).difference(table.column_names))


def _ok(dimension: Dimension, check: str, severity: Severity) -> CheckOutcome:
    return CheckOutcome(dimension=dimension, check=check, passed=True, severity=severity)


def _fail(
    dimension: Dimension, check: str, severity: Severity, message: str, failing_rows: int = 0
) -> CheckOutcome:
    return CheckOutcome(
        dimension=dimension, check=check, passed=False, severity=severity, message=message,
        failing_rows=failing_rows,
    )


def _array(table: pa.Table, column: str) -> pa.Array:
    return table.column(column).combine_chunks()


def _np_bool(values: pa.Array | pa.ChunkedArray, null_as: bool) -> np.ndarray:
    return np.asarray(pc.fill_null(values, null_as).to_numpy(zero_copy_only=False), dtype=bool)


def _is_list(dtype: pa.DataType) -> bool:
    return pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype)


def _list_depth(dtype: pa.DataType) -> tuple[int, pa.DataType]:
    depth = 0
    while _is_list(dtype):
        depth += 1
        dtype = dtype.value_type
    return depth, dtype


def _flatten(arr: pa.Array) -> tuple[pa.Array, np.ndarray, np.ndarray]:
    """→ (giá trị con, chỉ số dòng cha của từng giá trị con, vị trí của nó trong list cha)."""
    if len(arr) == 0:
        return pc.list_flatten(arr), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    parents = pc.list_parent_indices(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    values = pc.list_flatten(arr)
    lengths = np.asarray(pc.fill_null(pc.list_value_length(arr), 0).to_numpy(zero_copy_only=False), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    positions = np.arange(len(parents), dtype=np.int64) - starts[parents]
    return values, parents, positions


def _rows_with(n: int, rows: np.ndarray) -> np.ndarray:
    """Mảng bool độ dài n: True ở các dòng xuất hiện trong ``rows``."""
    return np.bincount(rows, minlength=n)[:n] > 0 if len(rows) else np.zeros(n, dtype=bool)


def _bad_values(values: pa.Array) -> np.ndarray:
    """NaN hoặc null."""
    bad = pc.is_null(values)
    if pa.types.is_floating(values.type):
        bad = pc.or_(bad, pc.fill_null(pc.is_nan(values), True))
    return _np_bool(bad, True)


def _leaf(arr: pa.Array, depth: int) -> tuple[pa.Array, np.ndarray]:
    """Giá trị lá của list lồng ``depth`` cấp + dòng gốc của từng giá trị."""
    rows = np.arange(len(arr), dtype=np.int64)
    for _ in range(depth):
        arr, parents, _ = _flatten(arr)
        rows = rows[parents]
    return arr, rows


# ──────────────────────────── Row masks (True = hợp lệ) ─────────────────────


def mask_not_null(table: pa.Table, config: NotNull) -> np.ndarray:
    ok = np.ones(table.num_rows, dtype=bool)
    for c in config.columns:
        ok &= _np_bool(pc.is_valid(table.column(c)), False)
    return ok


def mask_in_range(table: pa.Table, config: InRange) -> np.ndarray:
    col = table.column(config.column)
    bad = np.zeros(table.num_rows, dtype=bool)
    if config.min_value is not None:
        bad |= _np_bool(pc.less(col, config.min_value), False)
    if config.max_value is not None:
        bad |= _np_bool(pc.greater(col, config.max_value), False)
    return ~bad


def mask_positive(table: pa.Table, config: Positive) -> np.ndarray:
    ok = np.ones(table.num_rows, dtype=bool)
    for c in config.columns:
        ok &= ~_np_bool(pc.less_equal(table.column(c), 0), False)
    return ok


def mask_in_set(table: pa.Table, config: InSet) -> np.ndarray:
    col = table.column(config.column)
    allowed = pa.array(list(config.allowed), type=col.type)
    return _np_bool(pc.or_(pc.is_null(col), pc.is_in(col, value_set=allowed)), True)


def mask_list_length(table: pa.Table, config: ListLength) -> np.ndarray:
    lengths = pc.list_value_length(table.column(config.column))
    bad = _np_bool(pc.less(lengths, config.min_len), False)
    if config.max_len is not None:
        bad |= _np_bool(pc.greater(lengths, config.max_len), False)
    return ~bad


def mask_vector_shape(table: pa.Table, config: VectorShape) -> np.ndarray:
    n = table.num_rows
    arr = _array(table, config.column)
    ok = _np_bool(pc.is_valid(arr), False)
    level = arr
    rows = np.arange(n, dtype=np.int64)
    for size in config.shape:
        if not _is_list(level.type):
            return np.zeros(n, dtype=bool)
        lengths = _np_bool(pc.equal(pc.list_value_length(level), size), False)
        ok &= ~_rows_with(n, rows[~lengths])
        level, parents, _ = _flatten(level)
        rows = rows[parents]
    ok &= ~_rows_with(n, rows[_bad_values(level)])
    return ok


def mask_non_zero_vector(table: pa.Table, config: NonZeroVector) -> np.ndarray:
    n = table.num_rows
    arr = _array(table, config.column)
    ok = _np_bool(pc.is_valid(arr), False)
    depth, _ = _list_depth(arr.type)
    if depth == 1:
        values, rows = _leaf(arr, 1)
        nonzero = _np_bool(pc.not_equal(values, 0), False)
        return ok & _rows_with(n, rows[nonzero])
    if depth != 2:
        raise ValueError(f"NonZeroVector chỉ hỗ trợ vector 1 hoặc 2 chiều, {config.column} có {depth}")
    inner, inner_rows, inner_pos = _flatten(arr)
    values, value_inner, _ = _flatten(inner)
    nonzero = _np_bool(pc.not_equal(values, 0), False)
    inner_ok = _rows_with(len(inner), value_inner[nonzero])
    if config.rows is None:
        has_inner = _rows_with(n, inner_rows)
        return ok & has_inner & ~_rows_with(n, inner_rows[~inner_ok])
    wanted = np.isin(inner_pos, np.asarray(config.rows))
    hits = np.bincount(inner_rows[wanted & inner_ok], minlength=n)[:n] if n else np.zeros(0, dtype=np.int64)
    return ok & (hits == len(config.rows))


def mask_vector_range(table: pa.Table, config: VectorRange) -> np.ndarray:
    arr = _array(table, config.column)
    depth, _ = _list_depth(arr.type)
    values, rows = _leaf(arr, depth)
    inside = pc.and_(pc.greater_equal(values, config.min_value), pc.less_equal(values, config.max_value))
    bad = ~_np_bool(inside, False) | _bad_values(values)
    return ~_rows_with(table.num_rows, rows[bad])


_RELATION: dict[str, Callable[[pa.ChunkedArray, pa.ChunkedArray], pa.ChunkedArray]] = {
    ">=": pc.greater_equal,
    "<=": pc.less_equal,
    ">": pc.greater,
    "<": pc.less,
    "==": pc.equal,
    "!=": pc.not_equal,
}


def mask_column_relation(table: pa.Table, config: ColumnRelation) -> np.ndarray:
    if config.op not in _RELATION:
        raise ValueError(f"unsupported operator '{config.op}'")
    holds = _RELATION[config.op](table.column(config.left), table.column(config.right))
    return _np_bool(holds, True)  # một trong hai null → bỏ qua


def mask_foreign_key(table: pa.Table, config: ForeignKey) -> np.ndarray:
    col = table.column(config.column)
    known = config.reference.column(config.reference_column).combine_chunks()
    known = known.cast(col.type) if known.type != col.type else known
    return _np_bool(pc.or_(pc.is_null(col), pc.is_in(col, value_set=known)), True)


ROW_MASKS: dict[type, Callable[..., np.ndarray]] = {
    NotNull: mask_not_null,
    InRange: mask_in_range,
    Positive: mask_positive,
    InSet: mask_in_set,
    ListLength: mask_list_length,
    VectorShape: mask_vector_shape,
    NonZeroVector: mask_non_zero_vector,
    VectorRange: mask_vector_range,
    ColumnRelation: mask_column_relation,
    ForeignKey: mask_foreign_key,
}
ROW_CHECKS = tuple(ROW_MASKS)


def _columns_of(config: Check) -> tuple[str, ...]:
    if isinstance(config, ColumnRelation):
        return (config.left, config.right)
    if hasattr(config, "columns"):
        return tuple(config.columns)
    if hasattr(config, "column"):
        return (config.column,)
    return ()


def row_label(config: Check) -> str:
    """Tên ngắn của check, ghi vào cột ``_errors`` (vd ``non_zero_vector:text_emb[0]``)."""
    name = _NAMES[type(config)]
    if isinstance(config, ColumnRelation):
        return f"{name}:{config.left}{config.op}{config.right}"
    target = ",".join(_columns_of(config))
    if isinstance(config, NonZeroVector) and config.rows is not None:
        target += "[" + ",".join(map(str, config.rows)) + "]"
    return f"{name}:{target}"


def row_mask(table: pa.Table, config: Check) -> np.ndarray:
    try:
        fn = ROW_MASKS[type(config)]
    except KeyError:
        raise TypeError(f"{type(config).__name__} là check cấp lô, không có row_mask") from None
    missing = _missing_columns(table, _columns_of(config))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    return fn(table, config)


def split_valid(
    table: pa.Table, row_checks: Iterable[Check], error_column: str = "_errors"
) -> tuple[pa.Table, pa.Table]:
    """Tách dòng hợp lệ / dòng lỗi. ``bad`` có thêm cột ``error_column`` (list tên check bị vi phạm).

    Cột thiếu là lỗi cấp lô: chạy ``RequiredColumns`` trong suite cấp lô trước khi gọi hàm này.
    """
    checks = list(row_checks)
    n = table.num_rows
    errors: list[list[str]] = [[] for _ in range(n)]
    ok = np.ones(n, dtype=bool)
    for check in checks:
        mask = row_mask(table, check)
        label = row_label(check)
        for i in np.flatnonzero(~mask):
            errors[i].append(label)
        ok &= mask
    bad_idx = np.flatnonzero(~ok)
    bad = table.take(pa.array(bad_idx, type=pa.int64())).append_column(
        error_column, pa.array([errors[i] for i in bad_idx], type=pa.list_(pa.string()))
    )
    return table.filter(pa.array(ok)), bad


# ──────────────────────────── Check functions ──────────────────────────────


def _from_mask(table: pa.Table, config: Check, dimension: Dimension, message: str) -> CheckOutcome:
    name = _NAMES[type(config)]
    missing = _missing_columns(table, _columns_of(config))
    if missing:
        return _fail(dimension, name, config.severity, f"missing required columns: {missing}")
    try:
        bad = int((~ROW_MASKS[type(config)](table, config)).sum())
    except ValueError as exc:
        return _fail(dimension, name, config.severity, str(exc))
    if bad:
        return _fail(dimension, name, config.severity, f"{message} ({bad} row(s))", failing_rows=bad)
    return _ok(dimension, name, config.severity)


def check_required_columns(table: pa.Table, config: RequiredColumns) -> CheckOutcome:
    missing = _missing_columns(table, config.columns)
    if missing:
        return _fail(Dimension.COMPLETENESS, "required_columns", config.severity,
                     f"missing required columns: {missing}")
    return _ok(Dimension.COMPLETENESS, "required_columns", config.severity)


def check_not_null(table: pa.Table, config: NotNull) -> CheckOutcome:
    return _from_mask(table, config, Dimension.COMPLETENESS, f"{', '.join(config.columns)} contains null values")


def check_covers(table: pa.Table, config: Covers) -> CheckOutcome:
    if config.column not in table.column_names:
        return _fail(Dimension.COMPLETENESS, "covers", config.severity,
                     f"missing required columns: ['{config.column}']")
    expected = config.reference.column(config.reference_column).combine_chunks()
    present = table.column(config.column).combine_chunks().cast(expected.type)
    absent = int((~_np_bool(pc.is_in(expected, value_set=present), False)).sum())
    if absent:
        return _fail(Dimension.COMPLETENESS, "covers", config.severity,
                     f"{absent} {config.reference_column} của bảng tham chiếu không có trong {config.column}",
                     failing_rows=absent)
    return _ok(Dimension.COMPLETENESS, "covers", config.severity)


def check_unique(table: pa.Table, config: Unique) -> CheckOutcome:
    missing = _missing_columns(table, config.columns)
    if missing:
        return _fail(Dimension.UNIQUENESS, "unique", config.severity, f"missing required columns: {missing}")
    counts = table.select(list(config.columns)).group_by(list(config.columns)).aggregate([([], "count_all")])
    duplicated = int(_np_bool(pc.greater(counts.column("count_all"), 1), False).sum())
    if duplicated:
        return _fail(Dimension.UNIQUENESS, "unique", config.severity,
                     f"duplicate {' + '.join(config.columns)} rows", failing_rows=duplicated)
    return _ok(Dimension.UNIQUENESS, "unique", config.severity)


def check_in_range(table: pa.Table, config: InRange) -> CheckOutcome:
    return _from_mask(table, config, Dimension.VALIDITY,
                      f"{config.column} outside [{config.min_value}, {config.max_value}]")


def check_positive(table: pa.Table, config: Positive) -> CheckOutcome:
    return _from_mask(table, config, Dimension.VALIDITY, f"{', '.join(config.columns)} must be > 0")


def check_in_set(table: pa.Table, config: InSet) -> CheckOutcome:
    return _from_mask(table, config, Dimension.VALIDITY,
                      f"{config.column} has values outside {list(config.allowed)}")


def check_list_length(table: pa.Table, config: ListLength) -> CheckOutcome:
    return _from_mask(table, config, Dimension.VALIDITY,
                      f"{config.column} length outside [{config.min_len}, {config.max_len}]")


def check_vector_dtype(table: pa.Table, config: VectorDtype) -> CheckOutcome:
    if config.column not in table.column_names:
        return _fail(Dimension.VALIDITY, "vector_dtype", config.severity,
                     f"missing required columns: ['{config.column}']")
    dtype = table.schema.field(config.column).type
    depth, leaf = _list_depth(dtype)
    if depth != config.depth or not pa.types.is_floating(leaf):
        return _fail(Dimension.VALIDITY, "vector_dtype", config.severity,
                     f"{config.column} has type {dtype}, expected {config.depth}-level list of float")
    return _ok(Dimension.VALIDITY, "vector_dtype", config.severity)


def check_vector_shape(table: pa.Table, config: VectorShape) -> CheckOutcome:
    shape = "×".join(map(str, config.shape))
    return _from_mask(table, config, Dimension.VALIDITY, f"{config.column} not {shape} or contains NaN/null")


def check_non_zero_vector(table: pa.Table, config: NonZeroVector) -> CheckOutcome:
    return _from_mask(table, config, Dimension.VALIDITY, f"{row_label(config).split(':', 1)[1]} is a zero vector")


def check_vector_range(table: pa.Table, config: VectorRange) -> CheckOutcome:
    return _from_mask(table, config, Dimension.VALIDITY,
                      f"{config.column} has values outside [{config.min_value}, {config.max_value}]")


def check_column_relation(table: pa.Table, config: ColumnRelation) -> CheckOutcome:
    return _from_mask(table, config, Dimension.CONSISTENCY,
                      f"rule violated: {config.left} {config.op} {config.right}")


def check_foreign_key(table: pa.Table, config: ForeignKey) -> CheckOutcome:
    if config.reference_column not in config.reference.column_names:
        return _fail(Dimension.CONSISTENCY, "foreign_key", config.severity,
                     f"reference missing column '{config.reference_column}'")
    label = config.name or f"{config.column}->{config.reference_column}"
    return _from_mask(table, config, Dimension.CONSISTENCY, f"{label}: keys missing from reference")


def _means(table: pa.Table, column: str) -> np.ndarray:
    arr = _array(table, column)
    if _is_list(arr.type):
        values, _, positions = _flatten(arr)
        v = np.asarray(pc.fill_null(values.cast(pa.float64()), np.nan).to_numpy(zero_copy_only=False))
        keep = ~np.isnan(v)
        sums = np.bincount(positions[keep], weights=v[keep])
        counts = np.bincount(positions[keep])
        return sums / np.maximum(counts, 1)
    v = np.asarray(pc.fill_null(arr.cast(pa.float64()), np.nan).to_numpy(zero_copy_only=False))
    return np.array([np.nanmean(v)]) if (~np.isnan(v)).any() else np.array([])


def check_mean_shift(table: pa.Table, config: MeanShift) -> CheckOutcome:
    if config.column not in table.column_names:
        return _fail(Dimension.ACCURACY, "mean_shift", config.severity,
                     f"missing required columns: ['{config.column}']")
    ref = config.reference
    if ref is None or ref.num_rows == 0 or table.num_rows == 0 or config.column not in ref.column_names:
        return _ok(Dimension.ACCURACY, "mean_shift", config.severity)
    current, previous = _means(table, config.column), _means(ref, config.column)
    if current.shape != previous.shape:
        return _fail(Dimension.ACCURACY, "mean_shift", config.severity,
                     f"{config.column}: shape {current.shape} khác lô trước {previous.shape}")
    diff = float(np.max(np.abs(current - previous))) if current.size else 0.0
    if diff > config.max_abs_diff:
        return _fail(Dimension.ACCURACY, "mean_shift", config.severity,
                     f"{config.column}: mean lệch {diff:.3f} so với lô trước (ngưỡng {config.max_abs_diff})")
    return _ok(Dimension.ACCURACY, "mean_shift", config.severity)


def check_non_empty(table: pa.Table, config: NonEmpty) -> CheckOutcome:
    if table.num_rows == 0:
        return _fail(Dimension.TIMELINESS, "non_empty", config.severity, "batch is empty")
    return _ok(Dimension.TIMELINESS, "non_empty", config.severity)


def check_matches_date(table: pa.Table, config: MatchesDate) -> CheckOutcome:
    if config.column not in table.column_names:
        return _fail(Dimension.TIMELINESS, "matches_date", config.severity,
                     f"missing required columns: ['{config.column}']")
    as_text = pc.cast(table.column(config.column), pa.string())
    bad = int((~_np_bool(pc.equal(as_text, config.expected), True)).sum())
    if bad:
        return _fail(Dimension.TIMELINESS, "matches_date", config.severity,
                     f"{config.column} has rows outside expected date {config.expected}", failing_rows=bad)
    return _ok(Dimension.TIMELINESS, "matches_date", config.severity)


# ──────────────────────────── Dispatch + runner ────────────────────────────

_DISPATCH: dict[type, Callable[..., CheckOutcome]] = {
    RequiredColumns: check_required_columns,
    NotNull: check_not_null,
    Covers: check_covers,
    Unique: check_unique,
    InRange: check_in_range,
    Positive: check_positive,
    InSet: check_in_set,
    ListLength: check_list_length,
    VectorDtype: check_vector_dtype,
    VectorShape: check_vector_shape,
    NonZeroVector: check_non_zero_vector,
    VectorRange: check_vector_range,
    ColumnRelation: check_column_relation,
    ForeignKey: check_foreign_key,
    MeanShift: check_mean_shift,
    NonEmpty: check_non_empty,
    MatchesDate: check_matches_date,
}

_NAMES: dict[type, str] = {
    RequiredColumns: "required_columns",
    NotNull: "not_null",
    Covers: "covers",
    Unique: "unique",
    InRange: "in_range",
    Positive: "positive",
    InSet: "in_set",
    ListLength: "list_length",
    VectorDtype: "vector_dtype",
    VectorShape: "vector_shape",
    NonZeroVector: "non_zero_vector",
    VectorRange: "vector_range",
    ColumnRelation: "column_relation",
    ForeignKey: "foreign_key",
    MeanShift: "mean_shift",
    NonEmpty: "non_empty",
    MatchesDate: "matches_date",
}


def run_check(table: pa.Table, config: Check) -> CheckOutcome:
    """Chạy một check đơn lẻ, dispatch theo loại ``config``."""
    try:
        fn = _DISPATCH[type(config)]
    except KeyError:  # pragma: no cover - lập trình sai mới chạm tới
        raise TypeError(f"unknown check config: {type(config).__name__}") from None
    return fn(table, config)


def run_suite(table: pa.Table, checks: Iterable[Check], suite_name: str = "") -> ValidationResult:
    """Chạy cả suite cấp lô và gộp thành ``ValidationResult``.

    - ``errors``   : message của các check FAIL ở mức ERROR (lô không hợp lệ → dừng).
    - ``warnings`` : message của các check FAIL ở mức WARN (không chặn pipeline).
    """
    outcomes = tuple(run_check(table, check) for check in checks)
    prefix = f"{suite_name}: " if suite_name else ""
    errors = tuple(f"{prefix}{o.message}" for o in outcomes if not o.passed and o.severity is Severity.ERROR)
    warnings = tuple(f"{prefix}{o.message}" for o in outcomes if not o.passed and o.severity is Severity.WARN)
    return ValidationResult(is_valid=not errors, errors=errors, warnings=warnings, outcomes=outcomes)
