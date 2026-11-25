"""Test trên hạ tầng thật (`make up PROFILE=core`): Iceberg REST + MinIO + quyền bucket.

Đọc .env của repo (vicap.config gọi load_dotenv). Bỏ qua nếu dịch vụ chưa chạy.
"""

from __future__ import annotations

import urllib.request
import uuid

import pyarrow as pa
import pytest

pytestmark = pytest.mark.integration


def _up(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=3)
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def services() -> None:
    from vicap.config import IcebergConfig, MinioConfig

    if not (_up(f"{MinioConfig().endpoint}/minio/health/live") and _up(f"{IcebergConfig().uri}/v1/config")):
        pytest.skip("MinIO / Iceberg REST chưa chạy — make up PROFILE=core")


def test_ops_quarantine_on_rest_catalog(services) -> None:
    """Ghi bảng ops.quarantine qua REST catalog thật, ghi đè đúng phần (stage, batch_id)."""
    from vicap.iceberg.catalog import load_lakehouse_catalog
    from vicap.iceberg.reader import read_table
    from vicap.iceberg.tables import OPS_QUARANTINE
    from vicap.iceberg.writer import ensure_table
    from vicap.quality import write_quarantine_rows

    catalog = load_lakehouse_catalog()
    table = ensure_table(catalog, OPS_QUARANTINE.identifier, OPS_QUARANTINE.schema, OPS_QUARANTINE.partition_spec)
    batch = f"it-{uuid.uuid4().hex[:8]}"
    bad = pa.table({"video_id": ["1", "2"], "username": ["u", "u"], "_errors": [["e"], ["e"]]})
    common = {"stage": "integration_test", "batch_id": batch, "processing_date": "2026-11-25"}
    try:
        assert write_quarantine_rows(table, bad, **common) == 2
        assert write_quarantine_rows(table, bad.slice(0, 1), **common) == 1
        rows = read_table(table.refresh()).filter(pa.compute.equal(pa.compute.field("batch_id"), batch))
        assert rows.column("video_id").to_pylist() == ["1"]
        assert rows.column("raw_ref").to_pylist() == ["s3://raw/videos/1/"]
    finally:
        write_quarantine_rows(table.refresh(), bad.slice(0, 0), **common)


def test_bucket_permissions(services) -> None:
    """vicap-crawler ghi được raw/; vicap-pipeline chỉ đọc raw/, ghi được lakehouse/."""
    from vicap.config import MinioConfig, crawler_minio_config
    from vicap.storage import objects

    key = f"videos/it-{uuid.uuid4().hex[:8]}/video.mp4"
    crawler_fs = objects.s3_filesystem(crawler_minio_config())
    pipeline_fs = objects.s3_filesystem(MinioConfig())

    objects.put_bytes("raw", key, b"mp4", fs=crawler_fs)
    try:
        assert objects.get_bytes("raw", key, fs=pipeline_fs) == b"mp4"
        with pytest.raises(PermissionError):
            objects.put_bytes("raw", key, b"ghi-de", fs=pipeline_fs)
        with pytest.raises(PermissionError):
            objects.put_bytes("lakehouse", f"derived/{key}", b"x", fs=crawler_fs)
    finally:
        crawler_fs.rm_file(f"raw/{key}")
