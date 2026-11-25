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
def iceberg_scratch():
    """Tạo bảng trên REST catalog thật trong namespace tạm ``it_<hex>``; xoá bảng, namespace và file khi xong.

    Dùng cho test integration để không ghi vào bảng thật (``ops.quarantine``, ``bronze.*``…) và không để
    lại file trên ``lakehouse/warehouse/``. Gọi ``iceberg_scratch(name, table_def)`` → ``Table``.
    """
    from vicap.config import MinioConfig
    from vicap.iceberg.catalog import load_lakehouse_catalog
    from vicap.iceberg.writer import ensure_table
    from vicap.storage.objects import s3_filesystem

    catalog = load_lakehouse_catalog()
    namespace = f"it_{uuid4().hex[:8]}"
    created: list[str] = []

    def make(name: str, table_def):
        identifier = f"{namespace}.{name}"
        table = ensure_table(catalog, identifier, table_def.schema, table_def.partition_spec)
        created.append(identifier)
        return table

    try:
        yield make
    finally:
        for identifier in created:
            catalog.purge_table(identifier)
        if created:
            catalog.drop_namespace(namespace)
        config = MinioConfig()
        fs = s3_filesystem(config)
        prefix = f"{config.bucket}/warehouse/{namespace}"
        if fs.exists(prefix):
            fs.rm(prefix, recursive=True)


@pytest.fixture
def kafka_topic():
    """Topic tạm ``it.videos.<hex>`` (3 partition) trên Kafka thật; xoá khi xong. Bỏ qua nếu Kafka chưa chạy."""
    import socket

    from vicap.config import KafkaConfig

    host, port = KafkaConfig().bootstrap_servers.split(",")[0].rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
    except OSError:
        pytest.skip("Kafka chưa chạy — make up-crawl")
    from kafka.admin import KafkaAdminClient, NewTopic

    admin = KafkaAdminClient(bootstrap_servers=KafkaConfig().bootstrap_servers)
    topic = f"it.videos.{uuid4().hex[:8]}"
    admin.create_topics([NewTopic(topic, num_partitions=3, replication_factor=1)])
    try:
        yield topic
    finally:
        admin.delete_topics([topic])
        admin.close()


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
