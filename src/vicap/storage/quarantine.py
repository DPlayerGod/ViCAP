"""Ghi lô bị dừng vào ``s3://lakehouse/quarantine/`` (mức "dừng lô", PLAN mục 9.3), như HOSE.

``write_quarantine`` lưu ``data.parquet`` + ``errors.json`` để xem lại lô hỏng. Dòng bị cách ly (mức
"cách ly dòng") ghi vào bảng ``ops.quarantine`` bằng ``vicap.quality.quarantine``.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from vicap.config import MinioConfig
from vicap.storage import paths
from vicap.storage.objects import s3_filesystem


def _base(bucket: str, domain: str, processing_date: str, batch_id: str) -> str:
    return paths.s3_uri(bucket, paths.quarantine(domain, processing_date, batch_id))


def write_quarantine(
    table: pa.Table,
    errors: tuple[str, ...],
    *,
    domain: str,
    processing_date: str,
    batch_id: str,
    config: MinioConfig | None = None,
    bucket: str | None = None,
) -> str:
    """Ghi lô không hợp lệ và danh sách lỗi. Trả về URI thư mục đã ghi."""
    config = config or MinioConfig()
    fs = s3_filesystem(config)
    base = _base(bucket or config.bucket, domain, processing_date, batch_id)
    with fs.open(f"{base}/data.parquet", "wb") as f:
        pq.write_table(table, f)
    with fs.open(f"{base}/errors.json", "w") as f:
        json.dump(
            {"domain": domain, "processing_date": processing_date, "batch_id": batch_id, "errors": list(errors)},
            f,
            indent=2,
            ensure_ascii=False,
        )
    return base

