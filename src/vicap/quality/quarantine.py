"""Ghi dòng bị cách ly vào bảng Iceberg ``ops.quarantine`` (mức "cách ly dòng", PLAN mục 9.3).

Mỗi dòng ``bad`` của ``split_valid`` thành một dòng ``ops.quarantine``. Dạng bảng thay cho file Parquet
để đếm, lọc, join, đưa lên Superset và chạy lại dễ hơn. Lỗi cấp lô vẫn ghi file vào ``quarantine/`` trên
MinIO (``vicap.storage.quarantine``).

Cách dùng::

    res = validate(table, gold_features_suite(silver=silver_ids))
    res.batch.quarantine_and_raise(table, domain="gold_features", processing_date=ds, batch_id=batch_id)
    write_quarantine_rows(ops_table, res.bad, stage="gold_features", batch_id=batch_id, processing_date=ds)
    replace_by_keys(features_table, res.ok)
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import pyarrow as pa
from pyiceberg.expressions import And, EqualTo
from pyiceberg.table import Table

from vicap.iceberg.reader import read_table
from vicap.iceberg.writer import write_table
from vicap.storage import paths
from vicap.utils.dates import now_utc, parse_date

SEVERITIES = ("error", "warn")


def _column_or_null(bad: pa.Table, name: str) -> pa.Array:
    if name in bad.column_names:
        return bad.column(name).cast(pa.string()).combine_chunks()
    return pa.nulls(bad.num_rows, pa.string())


def quarantine_rows(
    bad: pa.Table,
    *,
    stage: str,
    batch_id: str,
    processing_date: str | date | datetime,
    severity: str = "error",
    raw_bucket: str = "raw",
    error_column: str = "_errors",
    created_at: datetime | None = None,
) -> pa.Table:
    """Dựng các dòng ``ops.quarantine`` từ ``bad`` (cần cột ``_errors``).

    ``raw_ref`` = ``s3://{raw_bucket}/videos/{video_id}/`` để mở lại file gốc; ``None`` khi dòng không có
    ``video_id`` (vd hồ sơ creator, khi đó chỉ có ``username``).
    """
    if severity not in SEVERITIES:
        raise ValueError(f"severity phải thuộc {SEVERITIES}: {severity!r}")
    if error_column not in bad.column_names:
        raise ValueError(f"thiếu cột {error_column!r} (lấy từ split_valid)")
    n = bad.num_rows
    video_ids = _column_or_null(bad, "video_id")
    raw_refs = [
        None if vid is None else paths.s3_uri(raw_bucket, paths.raw_prefix(vid)) for vid in video_ids.to_pylist()
    ]
    ts = created_at or now_utc()
    return pa.table({
        "video_id": video_ids,
        "username": _column_or_null(bad, "username"),
        "stage": pa.array([stage] * n, pa.string()),
        "batch_id": pa.array([batch_id] * n, pa.string()),
        "processing_date": pa.array([parse_date(processing_date)] * n, pa.date32()),
        "errors": bad.column(error_column).cast(pa.list_(pa.string())).combine_chunks(),
        "severity": pa.array([severity] * n, pa.string()),
        "raw_ref": pa.array(raw_refs, pa.string()),
        "resolved": pa.array([False] * n, pa.bool_()),
        "created_at": pa.array([ts] * n, pa.timestamp("us", tz="UTC")),
    })


def write_quarantine_rows(
    table: Table,
    bad: pa.Table,
    *,
    stage: str,
    batch_id: str,
    processing_date: str | date | datetime,
    severity: str = "error",
    raw_bucket: str = "raw",
) -> int:
    """Ghi đè các dòng của ``(stage, batch_id)`` trong ``ops.quarantine``; trả về số dòng đã ghi.

    Ghi đè theo ``(stage, batch_id)`` nên retry cùng ``batch_id`` không sinh dòng trùng. Lô không còn dòng
    lỗi thì xoá các dòng cũ của ``(stage, batch_id)``, để kết quả luôn khớp lần chạy mới nhất.
    """
    this_batch = And(EqualTo("stage", stage), EqualTo("batch_id", batch_id))
    if bad.num_rows == 0:
        table.delete(this_batch)
        return 0
    rows = quarantine_rows(
        bad, stage=stage, batch_id=batch_id, processing_date=processing_date, severity=severity,
        raw_bucket=raw_bucket,
    )
    write_table(table, rows, mode="overwrite", overwrite_filter=this_batch)
    return rows.num_rows


BAD_SCHEMA = pa.schema([("video_id", pa.string()), ("username", pa.string()), ("_errors", pa.list_(pa.string()))])


def merge_quarantine_rows(
    table: Table,
    bad: pa.Table,
    *,
    stage: str,
    batch_id: str,
    processing_date: str | date | datetime,
    resolved: Iterable[str] = (),
    severity: str = "error",
) -> int:
    """Như ``write_quarantine_rows`` nhưng gộp với dòng đã có của ``(stage, batch_id)`` thay vì thay hết.

    Dùng khi một lần chạy chỉ thấy một phần của lô (Bronze đọc Kafka theo offset: retry của crawl tới ở lần
    chạy sau). Dòng mới thay dòng cũ cùng ``video_id``; video trong ``resolved`` (lần này đã xử lý được) bị bỏ.
    Dòng không gắn video (``video_id`` null, vd hồ sơ creator) khoá theo ``(username, lỗi)``, ``resolved`` là username.
    Không có gì đổi thì không ghi. Trả về số dòng của ``(stage, batch_id)`` sau khi gộp.
    """
    resolved = set(resolved)
    old = read_table(table, And(EqualTo("stage", stage), EqualTo("batch_id", batch_id)))
    old_rows = [
        {"video_id": r["video_id"], "username": r["username"], "_errors": r["errors"]} for r in old.to_pylist()
    ]
    new_rows = bad.select(BAD_SCHEMA.names).cast(BAD_SCHEMA).to_pylist()

    def key(r: dict) -> object:
        return r["video_id"] if r["video_id"] is not None else (r["username"], tuple(r["_errors"]))

    def done(r: dict) -> bool:  # dòng không gắn video (hồ sơ creator): resolved chứa username
        return (r["video_id"] if r["video_id"] is not None else r["username"]) in resolved

    merged = {key(r): r for r in old_rows}
    merged.update({key(r): r for r in new_rows})
    rows = [r for r in merged.values() if not done(r)]
    if not new_rows and len(rows) == len(old_rows):
        return len(rows)
    return write_quarantine_rows(
        table, pa.Table.from_pylist(rows, schema=BAD_SCHEMA), stage=stage, batch_id=batch_id,
        processing_date=processing_date, severity=severity,
    )


def quarantine_by_batch(
    table: Table,
    problems: Iterable[tuple[str, str, str | None, str]],
    *,
    stage: str,
    processing_date: str | date | datetime,
    resolved: dict[str, set[str]] | None = None,
) -> None:
    """``problems``: ``(batch_id, video_id, username, lỗi)``. Gộp theo ``(stage, batch_id)`` bằng
    ``merge_quarantine_rows``; ``resolved[batch_id]``: video lần này xử lý được (bỏ khỏi quarantine)."""
    resolved = resolved or {}
    groups: dict[str, list[dict]] = {}
    for batch_id, video_id, username, error in problems:
        groups.setdefault(batch_id, []).append({"video_id": video_id, "username": username, "_errors": [error]})
    for batch_id in sorted(set(groups) | set(resolved)):
        merge_quarantine_rows(table, pa.Table.from_pylist(groups.get(batch_id, []), schema=BAD_SCHEMA),
                              stage=stage, batch_id=batch_id, processing_date=processing_date,
                              resolved=resolved.get(batch_id, ()))
