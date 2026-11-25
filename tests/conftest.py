from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    safe_name = request.node.name.replace("[", "_").replace("]", "_").replace(" ", "_")
    path = Path("data/test_outputs/pytest_tmp") / f"{safe_name}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


@pytest.fixture
def s3_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Biến môi trường tối thiểu để khởi tạo MinioConfig / IcebergConfig trong unit test."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")


@pytest.fixture
def sql_catalog(tmp_path: Path):
    """Catalog Iceberg thật trên sqlite + thư mục local, không cần Docker."""
    pytest.importorskip("sqlalchemy")
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path.resolve()
    return SqlCatalog(
        "test",
        uri=f"sqlite:///{warehouse / 'catalog.db'}",
        warehouse=warehouse.as_uri(),
        # PyArrowFileIO không mở được file:///C:/… trên Windows; fsspec thì được.
        **{"py-io-impl": "pyiceberg.io.fsspec.FsspecFileIO"},
    )
