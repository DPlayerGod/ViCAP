"""Đọc / ghi object trên MinIO theo bucket + key. Crawler và gpu-worker dùng chung.

Key dựng bằng ``vicap.storage.paths``. Truyền ``fs`` để dùng lại một kết nối khi ghi nhiều file.
"""

from __future__ import annotations

import s3fs

from vicap.config import MinioConfig


def s3_filesystem(config: MinioConfig | None = None) -> s3fs.S3FileSystem:
    config = config or MinioConfig()
    return s3fs.S3FileSystem(
        key=config.access_key,
        secret=config.secret_key,
        client_kwargs={"endpoint_url": config.endpoint, "region_name": config.region},
    )


def _path(bucket: str, key: str) -> str:
    if not bucket or not key or key.startswith("/"):
        raise ValueError(f"bucket / key không hợp lệ: {bucket!r}, {key!r}")
    return f"{bucket}/{key}"


def put_bytes(
    bucket: str,
    key: str,
    data: bytes,
    *,
    content_type: str | None = None,
    fs: s3fs.S3FileSystem | None = None,
) -> str:
    """Ghi đè ``s3://{bucket}/{key}``. Trả về URI."""
    fs = fs or s3_filesystem()
    kwargs = {"ContentType": content_type} if content_type else {}
    fs.pipe_file(_path(bucket, key), data, **kwargs)
    return f"s3://{bucket}/{key}"


def get_bytes(bucket: str, key: str, *, fs: s3fs.S3FileSystem | None = None) -> bytes:
    """Đọc cả object. Không có thì ``FileNotFoundError``."""
    return (fs or s3_filesystem()).cat_file(_path(bucket, key))


def exists(bucket: str, key: str, *, fs: s3fs.S3FileSystem | None = None) -> bool:
    return (fs or s3_filesystem()).exists(_path(bucket, key))
