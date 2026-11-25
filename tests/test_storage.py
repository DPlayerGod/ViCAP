"""vicap.storage.objects / quarantine. Unit test không cần MinIO; test S3 thật đánh dấu integration."""

from __future__ import annotations

import io
import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vicap.quality import ValidationResult
from vicap.storage import objects, paths
from vicap.storage.quarantine import write_quarantine


class _MemFS:
    """Đủ cho objects (pipe_file / cat_file / exists) và quarantine (open)."""

    def __init__(self) -> None:
        self.files: dict[str, object] = {}
        self.put_kwargs: dict[str, dict] = {}

    def pipe_file(self, path, data, **kwargs):
        self.files[path] = bytes(data)
        self.put_kwargs[path] = kwargs

    def cat_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def exists(self, path):
        return path in self.files

    @contextmanager
    def open(self, uri, mode="rb"):
        buf = io.BytesIO() if "b" in mode else io.StringIO()
        self.files[uri] = buf
        yield buf


class TestObjects:
    def test_put_get_exists(self) -> None:
        fs = _MemFS()
        key = paths.raw_video("123", paths.VIDEO_FILE)
        assert not objects.exists("raw", key, fs=fs)
        uri = objects.put_bytes("raw", key, b"mp4", content_type="video/mp4", fs=fs)
        assert uri == "s3://raw/videos/123/video.mp4"
        assert objects.exists("raw", key, fs=fs)
        assert objects.get_bytes("raw", key, fs=fs) == b"mp4"
        assert fs.put_kwargs["raw/videos/123/video.mp4"] == {"ContentType": "video/mp4"}

    def test_overwrite_and_missing(self) -> None:
        fs = _MemFS()
        objects.put_bytes("raw", "k", b"a", fs=fs)
        objects.put_bytes("raw", "k", b"b", fs=fs)
        assert objects.get_bytes("raw", "k", fs=fs) == b"b" and fs.put_kwargs["raw/k"] == {}
        with pytest.raises(FileNotFoundError):
            objects.get_bytes("raw", "missing", fs=fs)

    @pytest.mark.parametrize(("bucket", "key"), [("", "k"), ("raw", ""), ("raw", "/abs")])
    def test_rejects_bad_path(self, bucket: str, key: str) -> None:
        with pytest.raises(ValueError):
            objects.exists(bucket, key, fs=_MemFS())

    def test_default_fs_from_config(self, monkeypatch, s3_env) -> None:
        fs = _MemFS()
        monkeypatch.setattr("vicap.storage.objects.s3fs.S3FileSystem", lambda **_: fs)
        objects.put_bytes("raw", "k", b"x")
        assert fs.files == {"raw/k": b"x"}


@pytest.fixture
def fake_s3(monkeypatch, s3_env):
    fs = _MemFS()
    monkeypatch.setattr("vicap.storage.quarantine.s3_filesystem", lambda config=None: fs)
    return fs.files


class TestWriteQuarantine:
    def test_batch_quarantine(self, fake_s3) -> None:
        t = pa.table({"video_id": ["1"]})
        base = write_quarantine(t, ("e1", "lỗi 2"), domain="gold_features", processing_date="2026-11-25",
                                batch_id="b1")
        assert base == "s3://lakehouse/quarantine/gold_features/processing_date=2026-11-25/batch_id=b1"
        buf = fake_s3[f"{base}/data.parquet"]
        buf.seek(0)
        assert pq.read_table(buf).equals(t)
        meta = json.loads(fake_s3[f"{base}/errors.json"].getvalue())
        assert meta["errors"] == ["e1", "lỗi 2"] and meta["batch_id"] == "b1"


class TestQuarantineAndRaise:
    def test_noop_when_valid(self) -> None:
        ValidationResult(True, ()).quarantine_and_raise(
            pa.table({"a": [1]}), domain="d", processing_date="2026-11-25", batch_id="b"
        )

    def test_raises_with_uri(self, monkeypatch) -> None:
        monkeypatch.setattr("vicap.storage.quarantine.write_quarantine",
                            MagicMock(return_value="s3://lakehouse/quarantine/d"))
        with pytest.raises(ValueError, match=r"quarantined=s3://lakehouse/quarantine/d.*err_a; err_b"):
            ValidationResult(False, ("err_a", "err_b")).quarantine_and_raise(
                pa.table({"a": [1]}), domain="d", processing_date="2026-11-25", batch_id="b"
            )


@pytest.mark.integration
def test_s3_roundtrip() -> None:
    import urllib.request
    import uuid

    from vicap.config import MinioConfig

    config = MinioConfig()
    try:
        urllib.request.urlopen(f"{config.endpoint}/minio/health/live", timeout=3)
    except OSError:
        pytest.skip("MinIO chưa chạy — make up PROFILE=core")
    fs = objects.s3_filesystem(config)
    key = f"test_storage/{uuid.uuid4().hex[:8]}.bin"
    objects.put_bytes(config.bucket, key, b"\x00\x01", fs=fs)
    try:
        assert objects.exists(config.bucket, key, fs=fs)
        assert objects.get_bytes(config.bucket, key, fs=fs) == b"\x00\x01"
    finally:
        fs.rm_file(f"{config.bucket}/{key}")
